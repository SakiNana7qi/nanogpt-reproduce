import os
import time

save_dir = (
    f"/root/autodl-fs/gpt{time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))}"
)
os.makedirs(save_dir, exist_ok=True)
# os.system(f"cp -r /root/autodl-tmp/edu_fineweb10B {save_dir}/")
os.system(f"cp -r /root/gpt2-reproduce {save_dir}/")
