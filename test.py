import onnxruntime as ort
import numpy as np
import time

import os
import psutil
import threading

peak_rss = 0
def monitor_memory(pid, interval=0.01):
    global peak_rss
    process = psutil.Process(pid)
    while True:
        try:
            mem = process.memory_info().rss
            peak_rss = max(peak_rss, mem)
        except psutil.NoSuchProcess:
            break
        time.sleep(interval)
# 启动内存监控线程
monitor_thread = threading.Thread(target=monitor_memory, args=(os.getpid(),), daemon=True)
monitor_thread.start()

time.sleep(1)

# 打印最大内存
print(f"Peak RSS during inference: {peak_rss / 1024**2:.2f} MB")