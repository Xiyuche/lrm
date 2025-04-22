import torch
import torch.nn as nn
import torch.nn.functional as F
from .flash_attn_transformer import FlashTransformerEncoderLayer, FlashTransformerEncoder

class LaLRM(nn.Module):
    """
    Latent-based Large Reconstruction Model (LaLRM)

    根据附录 Fig. A4 所示的网络结构实现，包括：
      - 视频潜表示的 2D Conv patchify
      - 相机嵌入的 3D Conv patchify (原先与 Plücker embedding 类似，但这里直接保留 3D Conv 而已)
      - 拼接后线性映射到 1024 维
      - 24 层 Transformer 编码器
      - 3D 反卷积上采样得到 (B,12,T,H,W) 的特征
      - 最终 flatten 输出 (B, T*H*W, 12) 形式

    参数说明（可根据需要在构造函数中灵活设置）:
      upsample_st: 3D 转置卷积的 stride，区分低分辨率预训练或高分辨率微调 (例如 (4,16,16) 或 (4,8,8))
    """

    def __init__(
        self,
        upsample_st=(4, 8, 8),       # 反卷积的stride
        num_transformer_layers=24,   # Transformer层数
        d_model=1024,                # Transformer隐藏维度
        n_head=16,                   # Multi-head Attention的head数
        dim_feedforward=4096,        # FFN的维度
        use_flash_attn=True,         # 是否使用FlashAttention
        H_cam=480, W_cam=720         # 输出的分辨率（可能用于3D Conv的输入大小等）
    ):
        super().__init__()
        self.d_model = d_model
        self.H_cam = H_cam
        self.W_cam = W_cam

        # 1. 视频潜表示：Conv2D 做 spatial patchify
        #   in_c=16, out_c=1024, kernel=2, stride=2
        self.video_conv = nn.Conv2d(
            in_channels=16,
            out_channels=d_model,
            kernel_size=2,
            stride=2
        )
        self.video_ln = nn.LayerNorm(d_model)

        # 2. 相机嵌入：Conv3D 做时空 patchify
        #   in_c=6, out_c=1024, kernel=(4,16,16), stride=(4,16,16), pad=(2,0,0)
        self.camera_conv = nn.Conv3d(
            in_channels=6,
            out_channels=d_model,
            kernel_size=(4,16,16),
            stride=(4,16,16),
            padding=(2,0,0)
        )
        self.camera_ln = nn.LayerNorm(d_model)

        # 3. 拼接后 (2048 -> 1024) 的线性映射
        self.merge_linear = nn.Linear(d_model * 2, d_model)

        # 4. 根据标志决定使用 FlashAttention 或普通 Transformer
        if use_flash_attn:
            encoder_layer = FlashTransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                batch_first=True
            )
            self.transformer = FlashTransformerEncoder(
                encoder_layer,
                num_layers=num_transformer_layers
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_transformer_layers
            )

        # 5. 3D 反卷积 (in_c=1024, out_c=12, kernel=(5,8,8), stride=upsample_st, pad=(2,0,0))
        self.decode_3d = nn.ConvTranspose3d(
            in_channels=d_model,
            out_channels=12,
            kernel_size=(5,8,8),
            stride=upsample_st,
            padding=(2,0,0)
        )

        # 6. 最终可选激活处理 (Identity)
        self.final_act = nn.Identity()

    def generate_rays_from_camera(self, extrinsic: torch.Tensor, intrinsic: torch.Tensor,
                                  H: int, W: int):
        """
        从相机 extrinsic & intrinsic 生成每个像素对应的射线原点 (o) 和方向 (d).

        输入:
          extrinsic:  [B, N, 4, 4]   # 旋转+平移
          intrinsic:  [B, N, 4]      # (fx, fy, cx, cy)
          H, W:       图像分辨率

        输出:
          o, d: 分别为 [B, N, H, W, 3]
        """
        device = extrinsic.device
        B, N = extrinsic.shape[:2]

        # 创建像素网格: shape (H, W, 2)
        i_vals = torch.arange(W, device=device, dtype=torch.float32)
        j_vals = torch.arange(H, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(j_vals, i_vals, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

        # 扩展到 [B, N, H, W, 2]
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, N, H, W, 2)

        # 提取 fx, fy, cx, cy  => shape [B, N, 1, 1]
        fx = intrinsic[..., 0].unsqueeze(-1).unsqueeze(-1)
        fy = intrinsic[..., 1].unsqueeze(-1).unsqueeze(-1)
        cx = intrinsic[..., 2].unsqueeze(-1).unsqueeze(-1)
        cy = intrinsic[..., 3].unsqueeze(-1).unsqueeze(-1)

        # 转换到相机坐标系下
        x = (grid[..., 0] - cx) / fx
        y = (grid[..., 1] - cy) / fy
        z = torch.ones_like(x)
        dirs_camera = torch.stack([x, y, z], dim=-1)  # [B, N, H, W, 3]
        dirs_camera = F.normalize(dirs_camera, dim=-1)

        # 提取旋转R和平移t
        R = extrinsic[..., :3, :3]  # [B, N, 3, 3]
        t = extrinsic[..., :3, 3]   # [B, N, 3]

        # 旋转到世界坐标
        dirs_camera_flat = dirs_camera.view(B*N, H*W, 3).unsqueeze(-1)  # (B*N, H*W, 3, 1)
        R_flat = R.view(B*N, 3, 3)
        dirs_world_flat = torch.matmul(R_flat, dirs_camera_flat).squeeze(-1)
        dirs_world = dirs_world_flat.view(B, N, H, W, 3)
        d = F.normalize(dirs_world, dim=-1)

        # 将 t 扩展到像素分辨率
        o = t.view(B, N, 1, 1, 3).expand(-1, -1, H, W, -1)

        return o, d

    def forward_gaussian(self, video_latent: torch.Tensor, camera_embed: torch.Tensor,
                         extrinsics=None, intrinsics=None):
        """
        从 video_latent + camera_embed 中抽取 2D/3D patch, 拼接后送入 Transformer，
        反卷积得到形状 (B, T_out, H_out, W_out, 12) 的 gauss_points.

        可选地使用 extrinsics/intrinsics 来生成光线 o, d 与图像信息拼接。

        参数:
          video_latent: [B, T_v, H_v, W_v, C_v]
          camera_embed: [B, T_c, H_c, W_c, C_c]
          extrinsics:   [B, N, 4, 4], optional
          intrinsics:   [B, N, 4], optional

        返回:
          gaussian_points: [B, T_out, H_out, W_out, 12]
        """
        B, T_v, H_v, W_v, C_v = video_latent.shape

        # 如果需要用相机射线与图像合并:
        if (extrinsics is not None) and (intrinsics is not None):
            # o, d => [B, N, H_v, W_v, 3]  (这里 N 通常与 T_v 对应)
            o, d = self.generate_rays_from_camera(extrinsics, intrinsics, H_v, W_v)
            # Cross => (B, N, H_v, W_v, 3)
            cross_od = torch.cross(o, d, dim=-1)
            # 简单拼接 => shape [B, T_v, H_v, W_v, C_v+6]
            video_latent = torch.cat([video_latent, o, cross_od], dim=-1)

        # (B, T_v, H_v, W_v, C_v+...) => permute+view => (B*T_v, C_in, H_v, W_v)
        v_in = video_latent.permute(0,1,4,2,3).contiguous().view(B*T_v, -1, H_v, W_v)
        v_out = self.video_conv(v_in)   # => (B*T_v, d_model, H_v/2, W_v/2)
        v_out = v_out.flatten(2).permute(0,2,1).contiguous()  # => (B*T_v, N_spatial_v, d_model)
        v_out = self.video_ln(v_out)
        N_spatial_v = (H_v // 2) * (W_v // 2)
        v_out = v_out.view(B, T_v * N_spatial_v, -1)  # => (B, n_v, d_model)

        # camera_embed 同理 => (B, d_model, T_c', H_c', W_c')
        B_c, T_c, H_c, W_c, C_c = camera_embed.shape
        c_in = camera_embed.permute(0,4,1,2,3).contiguous()
        c_out = self.camera_conv(c_in)  # => (B, d_model, T_c', H_c', W_c')
        c_out = c_out.flatten(2).permute(0,2,1).contiguous()  # => (B, n_c, d_model)
        c_out = self.camera_ln(c_out)

        # 拼接 => (B, n_v + n_c, 2*d_model) => 线性映射 => (B, n_v + n_c, d_model)
        out = torch.cat([v_out, c_out], dim=-1)
        out = self.merge_linear(out)

        # 进入 Transformer => (B, n, d_model) => permute => (B, d_model, n)
        out = self.transformer(out)
        out = out.permute(0,2,1).contiguous()

        # 形状假设: out => (B, d_model, 13, 30, 45) (仅示例)
        out = out.view(B, self.d_model, 13, 30, 45)

        # 3D 反卷积 => (B, 12, T_out, H_out, W_out)
        out = self.decode_3d(out)
        out = self.final_act(out)

        # => (B, T_out, H_out, W_out, 12)
        gaussian_points = out.permute(0,2,3,4,1).contiguous()
        return gaussian_points

    def forward_render(self, gauss_points: torch.Tensor, render_camera_embd=None):
        """
        从 gauss_points + render_camera_embd 得到最终的 render_results
        shape 示例:
          gauss_points: [B, T_out, H_out, W_out, 12]
          render_camera_embd: [B, T_r, H_r, W_r, C_r], 仅示例

        可根据需要替换成真正的渲染操作
        """
        # 这里仍保持示例逻辑:
        render_results = gauss_points.mean(dim=-1, keepdim=True)  # 简单示例
        return render_results

    def forward(self,
                video_latent: torch.Tensor,
                camera_embed:  torch.Tensor,
                render_camera_embd=None,
                extrinsics=None,
                intrinsics=None):
        """
        主入口: 
          - 如果需要, 调用 generate_rays_from_camera 将相机射线信息与 video_latent 拼合
          - forward_gaussian 输出高斯体
          - forward_render 做最终渲染或可视化

        输入:
          video_latent:  [B, T_v, H_v, W_v, C_v]
          camera_embed:   [B, T_c, H_c, W_c, C_c]
          extrinsics:     [B, N, 4, 4], optional
          intrinsics:     [B, N, 4], optional
        """
        # 1) 生成高斯表征
        gauss_points = self.forward_gaussian(
            video_latent, camera_embed,
            extrinsics=extrinsics, intrinsics=intrinsics
        )

        # 2) 可选的渲染步骤
        render_results = self.forward_render(gauss_points, render_camera_embd)
        return render_results