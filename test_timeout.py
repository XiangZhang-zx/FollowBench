#!/usr/bin/env python3

import time
from datetime import timedelta
import torch
from accelerate import Accelerator, InitProcessGroupKwargs

def main():
    # 设置分布式超时为 5 秒（方便测试）
    kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=5))]
    accel = Accelerator(kwargs_handlers=kwargs)

    # 每个 rank 分别准备一个张量
    if accel.is_main_process:
        print("主进程睡眠 8 秒，会超时")
        t = torch.tensor([1]).to(accel.device)
        time.sleep(8)
        print("主进程继续")
    else:
        t = torch.tensor([1]).to(accel.device)
        print("其他进程等待 sync ...")
    
    try:
        accel.wait_for_everyone()  # 在这里会触发同步与可能的超时
        print("All synced OK on rank", accel.process_index)
    except Exception as e:
        print(f"Rank {accel.process_index} caught exception:", e)

if __name__ == "__main__":
    main()
