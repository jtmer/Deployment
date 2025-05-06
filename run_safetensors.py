import torch
import numpy as np
import time
import os
import psutil
import threading
from safetensors.torch import load_file

from TimerXL.models.timer_xl import Model
from TimerXL.models.configuration_timer import TimerxlConfig

# 内存监控线程
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

monitor_thread = threading.Thread(target=monitor_memory, args=(os.getpid(),), daemon=True)
monitor_thread.start()

# 模型初始化
model = Model(TimerxlConfig())
state_dict = load_file('model.safetensors')
state_dict = {'model.' + k: v for k, v in state_dict.items()}
missing, unexpected = model.load_state_dict(state_dict, strict=True)

device = 'cpu'
model.to(device=device)
model.eval()

# 构造 dummy 输入，假设输入形状是 [1, 672]
dummy_input = torch.rand(1, 672)

# 热身
for _ in range(3):
    _ = model(dummy_input)

# 正式推理计时
start_time = time.time()
with torch.no_grad():
    outputs = model(dummy_input)
end_time = time.time()

execution_time = (end_time - start_time) * 1000
print(f"Forward函数执行时间: {execution_time:.6f} 毫秒")
print(f"Peak RSS during inference: {peak_rss / 1024**2:.2f} MB")
