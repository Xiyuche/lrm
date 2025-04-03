import os
import objaverse
os.environ["OBJAVERSE_CACHE_DIR"] = "/mnt/public/yuchen.xi/OpenLRM/dataset/Objaverse-2"

uids = [
    "7f02efae906243f184a53997be60d9f4",
    "e6ea945cb96e44a9a426bd32a7754e29",
]

import random

random.seed(42)

uids = objaverse.load_uids()
random_object_uids = random.sample(uids, 100)
objects = objaverse.load_objects(uids=random_object_uids)

print("下载完成！自定义路径如下：")
for uid, path in objects.items():
    print(f"{uid}: {path}")