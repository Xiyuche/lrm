import torch
from modeling_lalrm import LaLRM  # 假设你把 LaLRM 放在 lalrm_model.py 里
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def check_model_memory(model, input_shapes, device='cuda'):
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)

    # 构造假的输入
    video_latent = torch.randn(*input_shapes['video'], device=device)
    camera_embed = torch.randn(*input_shapes['camera'], device=device)

    print("Running forward...")
    out = model(video_latent, camera_embed)

    print("Running backward...")
    loss = out.sum()
    loss.backward()

    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"\n[Memory Report] Peak allocated memory: {max_mem:.2f} GB")

    return max_mem

def main():
    # 假设我们做预训练阶段，使用 (13, 60, 90) 分辨率
    B = 2         # batch size
    T = 13        # 时间帧数
    H = 60        # 视频高度
    W = 90        # 视频宽度
    C = 16        # 视频 latent 通道数
    C_cam = 6    # 相机 Plücker 编码通道数

    # 构造假的 video latent（如来自 video diffusion encoder）
    video_latent = torch.randn(B, T, H, W, C).to(device)  # shape: (B, 13, 60, 90, 16)

    # 构造假的 camera embedding（Plücker 编码）
    # 原图尺寸是 480x720，下采样后约为 13x30x45
    T_cam = 49    # 相机帧数
    H_cam = 480   # 相机图像高度
    W_cam = 720   # 相机图像宽度
    camera_embed = torch.randn(B, T_cam, H_cam, W_cam, C_cam).to(device)

    # 构建模型（预训练用 upsample stride = (4,16,16)）
    model = LaLRM(upsample_st=(4,16,16)).to(device)

    # 前向传播
    output = model(video_latent, camera_embed)  # shape: (B, T_out * H_out * W_out, 12)

    # 打印输出形状检查
    print("LaLRM output shape:", output.shape)

    # 可选：检查每个维度
    B_out, N_points, feat_dim = output.shape
    print(f"Batch = {B_out}, Total points = {N_points}, Feature dim = {feat_dim}")

if __name__ == "__main__":
    # main()
    model = LaLRM(
        d_model=1024, 
        dim_feedforward=4096, 
        num_transformer_layers=24,
        # use_flash_attn=True,
    )

    input_shapes = {
        'video': (1, 13, 60, 90, 16),
        'camera': (1, 49, 480, 720, 6)
    }

    check_model_memory(model, input_shapes)