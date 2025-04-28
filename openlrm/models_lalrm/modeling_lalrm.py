import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
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

    def forward_gaussian(self, video_latent: torch.Tensor,
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

        # 计算相机射线
        # o, d => [B, N, H_v, W_v, 3]  (这里 N 通常与 T_v 对应)
        o, d = self.generate_rays_from_camera(extrinsics, intrinsics, H_v, W_v)
        # Cross => (B, N, H_v, W_v, 3)
        cross_od = torch.cross(o, d, dim=-1)
        # Get plücker embedding
        camera_embd = torch.cat([o, cross_od], dim=-1)  # [B, N, H_v, W_v, 9]


        # 简单拼接 => shape [B, T_v, H_v, W_v, C_v+6]
        video_latent = torch.cat([video_latent, o, cross_od], dim=-1)

        # (B, T_v, H_v, W_v, C_v+...) => permute+view => (B*T_v, C_in, H_v, W_v)
        v_in = video_latent.permute(0,1,4,2,3).contiguous().view(B*T_v, -1, H_v, W_v)
        v_out = self.video_conv(v_in)   # => (B*T_v, d_model, H_v/2, W_v/2)
        v_out = v_out.flatten(2).permute(0,2,1).contiguous()  # => (B*T_v, N_spatial_v, d_model)
        v_out = self.video_ln(v_out)
        N_spatial_v = (H_v // 2) * (W_v // 2)
        v_out = v_out.view(B, T_v * N_spatial_v, -1)  # => (B, n_v, d_model)


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
        gaussian_points = out.permute(0, 2, 3, 4, 1).contiguous()
        
        # 拆分最后一维为 distance, rgb, scaling, rotation, opacity
        distance, rgb, scaling, rotation, opacity = gaussian_points.split([1, 3, 3, 4, 1], dim=-1)

        # 将 ray distance 转换为 xyz
        w = torch.sigmoid(distance)
        near, far = 0.1, 10.0  # 假设的 near 和 far 平面距离
        xyz = o + d * (near * (1 - w) + far * w)

        # 对 scaling 和 rotation 进行处理
        scaling = torch.clamp(torch.exp(scaling - 2.3), max=0.3)
        rotation = rotation / rotation.norm(dim=-1, keepdim=True)
        opacity = torch.sigmoid(opacity - 2.0)

        # 返回 xyz, rgb, scaling, rotation, opacity
        return xyz, rgb, scaling, rotation, opacity


    def render_in_target_camera(
        self,
        means3D:       torch.Tensor,  # [B, N, 3]
        rotations:     torch.Tensor,  # [B, N, 4]
        scales:        torch.Tensor,  # [B, N, 3]
        opacities:     torch.Tensor,  # [B, N, 1]
        colors_precomp:torch.Tensor,  # [B, N, C], 如 RGB=3 或 SH展开后更多通道
        K:             torch.Tensor,  # [B,   4, 4], 每个batch对应一个相机内参矩阵(含fx,fy,cx,cy等)
        RT:            torch.Tensor,  # [B,   4, 4], 每个batch对应一个旋转平移外参
        width:         int   = 256,
        height:        int   = 256,
        sh_degree:     int   = 0,     # 设为0表示无Spherical Harmonics或仅RGB
        near_plane:    float = 0.00001,
        radius_clip:   float = 0.1,
        render_mode:   str   = "RGB+D"
    ):
        """
        使用 gsplat.rendering.rasterization 在指定相机参数 (K, RT) 下进行批量渲染，
        返回 render_colors, render_alphas, render_depths (以及其他你需要的中间结果).

        参数:
        means3D:       (B, N, 3)  高斯中心/Mean
        rotations:     (B, N, 4)  高斯旋转的四元数
        scales:        (B, N, 3)  高斯缩放/Scale
        opacities:     (B, N, 1)  每个高斯的透明度
        colors_precomp:(B, N, C)  颜色或其他SH通道 (若 C=3 则仅RGB)
        K:             (B, 4, 4)  相机内参矩阵 (fx, fy, cx, cy 在对角 & .[0,2])
        RT:            (B, 4, 4)  相机外参矩阵
        width, height: 渲染图像分辨率
        sh_degree:     若为0则无SH展开，若>0则表示颜色通道是Spherical Harmonics
        near_plane:    最近裁剪平面
        render_mode:   默认为 \"RGB+D\"，可输出颜色+深度
        radius_clip:   光栅化时对高斯半径的截断

        返回:
        dict，包含三张图:
            \"image\" -> (B, 3, H, W)
            \"alpha\" -> (B, 1, H, W)
            \"depth\" -> (B, 1, H, W)
        """

        # 注意：如果 colors_precomp.shape[-1]==3，可视为普通RGB，否则视为多通道(例如 SH展开)
        # 因为我们在调用rasterization时可以一次性传入所有batch，下面直接传 B维度

        # (B, 4, 4) => 取 [:, :3, :3] 得到 (B, 3, 3) 旋转、保留 fx,fy,cx,cy
        # 也可以不改，取 K[:, :3, :3] 传给Ks
        # 同理 RT => viewmats = RT.inverse()
        # 如果 batch_size>1，可以一次性把K, RT 全部传进去

        # squeeze(-1): (B, N, 1)->(B, N)
        opacities_squeezed = opacities.squeeze(dim=-1)

        # 直接一次性处理:
        render_colors, render_alphas, meta = rasterization(
            means       = means3D,             # (B, N, 3)
            quats       = rotations,           # (B, N, 4)
            scales      = scales,              # (B, N, 3)
            opacities   = opacities_squeezed,  # (B, N)
            colors      = colors_precomp,      # (B, N, C), C=3 或 (SH相关)
            sh_degree   = sh_degree,           # Spherical Harmonics次数
            viewmats    = RT.inverse(),        # (B, 4, 4) => inverse() => (B, 4, 4)
            Ks          = K[:, :3, :3],        # (B, 3, 3)
            width       = width,
            height      = height,
            near_plane  = near_plane,
            render_mode = render_mode,
            radius_clip = radius_clip
            # 可能还可指定 rasterize_mode="antialiased" 等
        )
        # 默认 rasterization 输出:
        #  render_colors: (B, H, W, channels=4?) 取决于 render_mode
        #  render_alphas: (B, H, W, 4) 也可能是(可与render_colors匹配)
        #  meta: 其他可能的信息

        # 下面演示如何解析: 例如在 \"RGB+D\" 模式下 render_colors[..., :3] 是RGB, [:, :, 3] 是Depth
        # alpha 通常在 render_alphas[..., 3] or 取 single-channel
        # 你也可以根据 rasterization 的实际输出格式做调整

        # permute => (B, channel, H, W)
        # 拿到 RGBA(D) 第4维通道
        # 例如 shape: (B, H, W, 4) => permute => (B, 4, H, W)
        render_colors_4d = render_colors.permute(0, 3, 1, 2).contiguous()  # => (B, 4, H, W)
        # 取前三通道 = RGB
        rendered_rgb   = render_colors_4d[:, 0:3, :, :]  # (B, 3, H, W)
        # 取第四通道 = Depth or A 需看render_mode
        rendered_depth = render_colors_4d[:, 3:4, :, :]  # (B, 1, H, W)

        # alpha 类似 => (B, H, W, 4) => permute => (B, 4, H, W)
        # 具体看 rasterization 的 actual shape
        render_alphas_4d = render_alphas.permute(0, 3, 1, 2).contiguous()  # => (B, 4, H, W)
        # 这里假定 alpha 在第0通道或最后一个通道:
        rendered_alpha = render_alphas_4d[:, 0:1, :, :]  # or [:, 3:4, :, :]

        # 返回一个字典
        return {
            "image": rendered_rgb,      # (B, 3, H, W)
            "alpha": rendered_alpha,    # (B, 1, H, W)
            "depth": rendered_depth,    # (B, 1, H, W)
            "meta":  meta               # optional
        }

    def forward(self,
                video_latent: torch.Tensor,
                target_camera=None,
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
          extrinsics:     [B, N, 4, 4]
          intrinsics:     [B, N, 4]
     """

        xyz, rgb, scaling, rotation, opacity = self.forward_gaussian(
            video_latent,
            extrinsics=extrinsics, intrinsics=intrinsics
        )


        render_results = self.render_in_target_camera(
            means3D=xyz,
            rotations=rotation,
            scales=scaling,
            opacities=opacity,
            colors_precomp=rgb,
            K=target_camera[:, :3, :3],
            RT=target_camera,
            width=self.W_cam, height=self.H_cam
        )

        return render_results