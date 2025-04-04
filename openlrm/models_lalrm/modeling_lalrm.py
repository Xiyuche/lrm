import torch
import torch.nn as nn
import torch.nn.functional as F

class LaLRM(nn.Module):
    """
    Latent-based Large Reconstruction Model (LaLRM)

    根据附录 Fig. A4 所示的网络结构实现，包括：
      - 视频潜表示的 2D Conv patchify
      - 相机 Plücker 嵌入的 3D Conv patchify
      - 拼接后线性映射到 1024 维
      - 24 层 Transformer 编码器
      - 3D 反卷积上采样得到 (B,12,T,H,W) 的特征
      - 最终 flatten 输出 (B, T*H*W, 12) 形式

    参数说明（可根据需要在构造函数中灵活设置）:
      upsample_st: 3D 转置卷积的 stride，预训练/微调可不同（(4,16,16) 或 (4,8,8)）。
    """
    def __init__(
        self,
        upsample_st=(4, 8, 8),       # 反卷积的stride，区分低分辨率预训练或高分辨率微调
        num_transformer_layers=24,   # Transformer层数
        d_model=1024,                # Transformer隐藏维度
        n_head=16,                   # Multi-head Attention的head数
        dim_feedforward=4096         # FFN的维度
    ):
        super().__init__()

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
        #   in_c=16, out_c=1024, kernel=(4,16,16), stride=(4,16,16), pad=(2,0,0)
        self.camera_conv = nn.Conv3d(
            in_channels=16, 
            out_channels=d_model, 
            kernel_size=(4,16,16),
            stride=(4,16,16),
            padding=(2,0,0)
        )
        self.camera_ln = nn.LayerNorm(d_model)

        # 3. 拼接后 (2048 -> 1024) 的线性映射
        self.merge_linear = nn.Linear(d_model * 2, d_model)

        # 4. Transformer 编码器，24层、隐藏维度1024，可视需要替换成FlashAttention
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

        # 6. 可进一步对输出特征做激活（如对 density/颜色分块处理），此处暂留空或Identity
        self.final_act = nn.Identity()

    def forward(self, video_latent, camera_embed):
        """
        参数:
          video_latent: 形状 (B, T_v, H_v, W_v, 16)，例如 (B,13,60,90,16) 对应低分辨率
          camera_embed: 形状 (B, T_c, H_c, W_c, 16)，例如 (B,49,480,720,16)
        
        返回:
          gauss_points: (B, T_c * H_out * W_out, 12)，
                        其中 T_c*H_out*W_out 常见形如 49*240*360 = 4,233,600
        """

        B, T_v, H_v, W_v, C_v = video_latent.shape
        # (B, T_v, H_v, W_v, C_v) -> (B*T_v, C_v, H_v, W_v)
        v_in = video_latent.permute(0,1,4,2,3).contiguous().view(B*T_v, C_v, H_v, W_v)
        # 2D卷积下采样 (kernel=2, stride=2)
        v_out = self.video_conv(v_in)  # (B*T_v, 1024, H_v/2, W_v/2)
        # flatten + permute  => (B*T_v, N_spatial, 1024)
        v_out = v_out.flatten(2).permute(0,2,1).contiguous()
        # 每个时刻分开 LayerNorm 后再合并
        v_out = self.video_ln(v_out)
        # 复原回 (B, T_v*N_spatial, 1024)
        N_spatial_v = (H_v // 2) * (W_v // 2)
        v_out = v_out.view(B, T_v * N_spatial_v, -1)  # => (B, n_v, d_model)

        # 相机 3D卷积
        B_c, T_c, H_c, W_c, C_c = camera_embed.shape
        c_in = camera_embed.permute(0,4,1,2,3).contiguous()  # (B, 16, T_c, H_c, W_c)
        c_out = self.camera_conv(c_in)  # (B, 1024, T_c', H_c', W_c')
        c_out = c_out.flatten(2).permute(0,2,1).contiguous()  # => (B, n_c, 1024)
        c_out = self.camera_ln(c_out)  # => (B, n_c, 1024)

        # 拼接 => (B, n, 2048) => linear => (B, n, 1024)
        out = torch.cat([v_out, c_out], dim=-1)
        out = self.merge_linear(out)  # => (B, n, d_model)

        # 送入 Transformer
        out = self.transformer(out)   # => (B, n, d_model)

        # reshape 成 (B, d_model, T_c', H_c', W_c')
        # 注意：n == T_c'*H_c'*W_c'，需和 camera_conv 输出匹配
        # 假设 T_c'=13, H_c'=30, W_c'=45（对应下采样后）
        # 也可根据实际分辨率算出 T_c', H_c', W_c'
        out = out.permute(0,2,1).contiguous()
        # 假设下游形状固定为 (B, 1024, 13,30,45)
        # 若做通用实现可在外部传参或者根据 camera_embed 动态计算
        out = out.view(B, 1024, 13, 30, 45)

        # 3D 反卷积 => (B,12, T_out, H_out, W_out)
        out = self.decode_3d(out)

        # 可选激活
        out = self.final_act(out)

        # 输出的最后一个通道维度 (12) 表示每个高斯点的属性:
        # [0:3]   → RGB颜色
        # [3:6]   → 3D尺度（scale_x, scale_y, scale_z）
        # [6:10]  → 旋转四元数 (quaternion: x, y, z, w)
        # [10]    → 不透明度（opacity）
        # [11]    → 相机光线距离（ray distance）
        # 调整到 (B, T_out, H_out, W_out, 12)，再 flatten => (B, T_out*H_out*W_out, 12)
        out = out.permute(0,2,3,4,1).contiguous()
        B2, T2, H2, W2, C2 = out.shape
        gauss_points = out.view(B2, T2*H2*W2, C2)

        return gauss_points