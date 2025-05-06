import onnxruntime as ort
import numpy as np
import time
import torch

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

session = ort.InferenceSession("model_int8.onnx", providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_dtype = session.get_inputs()[0].type  # 通常是 'tensor(float)' 等

print(f"输入名称: {input_name}, 输入形状: {input_shape}, 输入类型: {input_dtype}")

dummy_input = np.random.rand(16000, 672).astype(np.float32)

for i in range(3):
    outputs = session.run(None, {input_name: dummy_input})

start_time = time.time()
outputs = session.run(None, {input_name: dummy_input})
end_time = time.time()

execution_time = (end_time - start_time) * 1000  # 转成毫秒
print(f"Forward函数执行时间: {execution_time:.6f} 毫秒")

# 打印最大内存
print(f"Peak RSS during inference: {peak_rss / 1024**2:.2f} MB")