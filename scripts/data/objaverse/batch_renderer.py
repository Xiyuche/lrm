import os, json, subprocess
from multiprocessing import Pool

# 配置参数
BLENDER_EXEC = "/mnt/public/yuchen.xi/app/blender-4.0.0-linux-x64/blender"
SCRIPT_PATH = "scripts/data/objaverse/blender_script.py"
INPUT_DIR = "dataset/Objaverse/glbs/000-000"
OUTPUT_DIR = "dataset/Objaverse/rendered"
UID_LIST_PATH = os.path.join(OUTPUT_DIR, "uids.json")

NUM_GPUS = 8
WORKERS_PER_GPU = 20
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

# 获取所有 glb 文件路径
glb_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".glb")]
glb_paths = [os.path.join(INPUT_DIR, f) for f in glb_files]

def render_task(task):
    idx, glb_path = task
    uid = os.path.basename(glb_path).split(".")[0]
    gpu_id = idx % NUM_GPUS  # 分配 GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        BLENDER_EXEC, "-b", "-P", SCRIPT_PATH, "--",
        "--object_path", glb_path,
        "--output_dir", OUTPUT_DIR,
        "--engine", "CYCLES",
        "--num_images", "8",
        "--resolution", "512"
    ]

    try:
        subprocess.run(cmd, check=True, env=env)
        return uid
    except subprocess.CalledProcessError:
        print(f"❌ Failed: {uid}")
        return None

if __name__ == "__main__":
    print(f"🔧 Starting {TOTAL_WORKERS} workers across {NUM_GPUS} GPUs...")
    with Pool(processes=TOTAL_WORKERS) as pool:
        results = pool.map(render_task, list(enumerate(glb_paths)))

    # 保存成功的 uid
    uids = [uid for uid in results if uid]
    with open(UID_LIST_PATH, "w") as f:
        json.dump(uids, f, indent=2)
    print(f"✅ Done. Successfully rendered {len(uids)} models.")