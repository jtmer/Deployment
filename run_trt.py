import tensorrt as trt

import os
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import torch
torch.randn(1).cuda()

import threading
import time
import subprocess

peak_mem_mb = 0

def monitor_gpu_memory(pid=None, interval=0.1, gpu_id=0):
    global peak_mem_mb
    while True:
        try:
            # 查询指定 GPU 上的显存使用（MiB）
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_id}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                proc_pid, mem_used = map(str.strip, line.split(','))
                if pid is None or int(proc_pid) == pid:
                    mem = int(mem_used)
                    peak_mem_mb = max(peak_mem_mb, mem)
        except Exception as e:
            print("Memory monitor error:", e)
            break
        time.sleep(interval)

# 使用方法：
# 启动监控线程（可提前几秒开始）
monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(os.getpid(), 0.001, 0), daemon=True)
monitor_thread.start()

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path), f"Engine file not found: {engine_file_path}"
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    total_memory = 0

    for binding_idx in range(engine.num_io_tensors):
        binding = engine.get_tensor_name(binding_idx)
        shape = engine.get_tensor_shape(binding)
        
        # 修复动态形状处理
        # 使用最大批量大小计算缓冲区
        fixed_shape = list(shape)
        if shape[0] == -1:  # 如果批量维度是动态的
            fixed_shape[0] = 160  # 使用足够大的批量大小，根据实际需求调整
            print(f"Dynamic batch detected, allocating buffers for batch size {fixed_shape[0]}")
        
        # 使用修正后的形状计算大小
        size = np.prod(fixed_shape)  # 使用numpy函数而不是trt.volume()
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))
        
        memory_bytes = size * dtype.itemsize
        total_memory += memory_bytes
        
        print(f"Tensor: {binding}, Original shape: {shape}, Fixed shape: {fixed_shape}, Size: {size}, Memory: {memory_bytes/1024/1024:.2f} MB")
        
        try:
            host_mem = cuda.pagelocked_empty(int(size), dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))
        except Exception as e:
            print(f"Error allocating memory for {binding}: {e}")
            print(f"Try reducing batch size or sequence length")
            raise
            
    print(f"Total memory needed: {total_memory/1024/1024:.2f} MB")
    return inputs, outputs, bindings, stream


def infer(context, bindings, inputs, outputs, stream, input_data):
    """使用TensorRT引擎进行推理"""
    host_input, device_input = inputs[0]
    host_output, device_output = outputs[0]
    
    # 打印输入数据信息
    print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")
    print(f"Input sample values: {input_data.ravel()[:5]}")
    
    # 确保数据类型是float32
    if input_data.dtype != np.float32:
        input_data = input_data.astype(np.float32)
    
    # 动态形状处理 - 使用API兼容性包装
    try:
        # 获取输入和输出索引
        input_idx = 0
        output_idx = context.engine.num_io_tensors - 1
        
        # 获取输入和输出名称
        input_name = context.engine.get_tensor_name(input_idx)
        output_name = context.engine.get_tensor_name(output_idx)
        
        print(f"Setting shape for {input_name} to {input_data.shape}")
        
        # 尝试各种可能的API
        try:
            # 新API (TensorRT 8.x+)
            context.set_input_shape(input_name, input_data.shape)
            print("Used set_input_shape")
        except AttributeError:
            try:
                # 旧API
                context.set_binding_shape(input_idx, input_data.shape)
                print("Used set_binding_shape")
            except AttributeError:
                print("Dynamic shape setting not supported in this TensorRT version")
        
        # 获取输出形状
        try:
            output_shape = context.get_tensor_shape(output_name)
            print("Used get_tensor_shape")
        except AttributeError:
            try:
                output_shape = context.get_binding_shape(output_idx)
                print("Used get_binding_shape")
            except AttributeError:
                print("Could not determine output shape")
                output_shape = None
        
        if output_shape:
            print(f"Output shape: {output_shape}")
        
        # 验证缓冲区大小是否足够
        input_size = np.prod(input_data.shape)
        if input_size > host_input.size:
            print(f"Warning: Input size {input_size} exceeds buffer size {host_input.size}")
            raise ValueError(f"Input buffer too small for data shape {input_data.shape}")
    except Exception as e:
        print(f"Error during dynamic shape handling: {e}")
        print("Trying alternative approach...")
    
    # 将输入数据复制到主机内存
    np.copyto(host_input, input_data.ravel())
    
    # 将数据从主机内存复制到设备内存
    cuda.memcpy_htod_async(device_input, host_input, stream)
    
    # 执行推理
    success = False
    
    # 尝试所有可能的执行方法
    execution_methods = [
        {
            'name': 'execute_v2',
            'fn': lambda: context.execute_v2(bindings=bindings)
        },
        {
            'name': 'execute_async_v3',
            'fn': lambda: context.execute_async_v3(stream_handle=stream.handle)
        },
        {
            'name': 'execute_async_v2',
            'fn': lambda: context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        },
        {
            'name': 'execute_async',
            'fn': lambda: context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
        },
        {
            'name': 'execute',
            'fn': lambda: context.execute(batch_size=1, bindings=bindings)
        }
    ]
    
    for method in execution_methods:
        try:
            print(f"Trying {method['name']}...")
            method['fn']()
            print(f"{method['name']} succeeded!")
            success = True
            break
        except (AttributeError, RuntimeError) as e:
            print(f"{method['name']} failed: {e}")
    
    # 复制输出数据从设备到主机
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    
    # 同步流
    stream.synchronize()
    
    if not success:
        print("WARNING: All execution methods failed")
    
    # 检查输出是否全为零
    if np.all(host_output == 0):
        print("WARNING: Output is all zeros, inference likely failed!")
    else:
        print("Inference successful, output contains non-zero values")
    
    print(f"Output sample: {host_output[:5]}")
    
    return host_output

def infer_with_latest_api(engine, context, input_data):
    """使用最新的TensorRT API进行推理"""
    # 创建CUDA流
    stream = cuda.Stream()
    
    # 分配主机和设备内存
    input_shape = list(input_data.shape)
    
    # 获取输入和输出名称
    input_idx = 0
    output_idx = engine.num_io_tensors - 1
    input_name = engine.get_tensor_name(input_idx)
    output_name = engine.get_tensor_name(output_idx)
    
    # 设置输入形状 - 正确处理API差异
    try:
        context.set_input_shape(input_name, input_shape)
        print("Used set_input_shape")
    except AttributeError:
        try:
            context.set_binding_shape(input_idx, input_shape)
            print("Used set_binding_shape")
        except AttributeError:
            print("Cannot set dynamic shape with current TensorRT version")
    
    # 获取输出形状
    try:
        output_shape = context.get_tensor_shape(output_name)
        print("Used get_tensor_shape")
    except AttributeError:
        try:
            output_shape = context.get_binding_shape(output_idx)
            print("Used get_binding_shape")
        except AttributeError:
            print("Could not determine output shape")
            # 使用引擎中的静态形状作为后备
            output_shape = engine.get_tensor_shape(output_name)
    
    print(f"Input shape: {input_shape}, Output shape: {output_shape}")
    
    # 分配适当大小的缓冲区
    input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(input_name)))
    output_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(output_name)))
    
    # 分配主机内存
    h_input = cuda.pagelocked_empty(np.prod(input_shape), input_dtype)
    h_output = cuda.pagelocked_empty(np.prod(output_shape), output_dtype)
    
    # 分配设备内存
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    # 创建绑定列表
    bindings = [int(d_input), int(d_output)]
    
    # 复制输入数据到主机内存
    np.copyto(h_input, input_data.ravel())
    
    # 将数据从主机复制到设备
    cuda.memcpy_htod_async(d_input, h_input, stream)
    
    # 执行推理 - 尝试不同的API
    success = False
    try:
        # 首选非异步版本，因为我们会等待流同步
        context.execute_v2(bindings=bindings)
        success = True
        print("Used execute_v2")
    except AttributeError:
        try:
            context.execute(batch_size=1, bindings=bindings)
            success = True
            print("Used execute")
        except AttributeError:
            try:
                # 如果以上都失败，尝试异步版本
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                success = True
                print("Used execute_async_v2")
            except AttributeError:
                context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
                success = True
                print("Used execute_async")
    
    # 将结果从设备复制到主机
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    
    # 同步流
    stream.synchronize()
    
    # 将输出重塑为正确的形状
    output = h_output.reshape(output_shape)
    
    # 检查输出是否全为零
    if np.all(output == 0) and success:
        print("WARNING: Output is all zeros despite successful execution!")
    elif success:
        print("Inference successful with non-zero outputs")
        
    print(f"Output sample: {output.ravel()[:5]}")
    
    return output

def count_csv_rows(file_path):
    with open(file_path, 'r') as f:
        # 读取第一行（标题）
        header = f.readline()
        # 计算剩余行数
        count = sum(1 for _ in f)
    return count + 1  # 加上标题行

if __name__ == "__main__":
    # 加载完整数据集
    import pandas as pd
    
    dummy_input = np.random.rand(160, 672).astype(np.float32)
    
    # 加载引擎
    print("Loading TensorRT engine...")
    engine = load_engine("model_int8.engine")
    context = engine.create_execution_context()
    
    # 分配缓冲区(只需分配一次)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    for i in range(3):
        # 使用标准方法
        int8_output = infer(context, bindings, inputs, outputs, stream, dummy_input)
        
        # 如果输出全为0，使用备用方法
        if np.all(int8_output == 0):
            print("Standard inference failed, trying alternative method...")
            int8_output = infer_with_latest_api(engine, context, dummy_input)
    

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    # 使用标准方法
    for i in range(20):
        int8_output = infer(context, bindings, inputs, outputs, stream, dummy_input)
        
        # 如果输出全为0，使用备用方法
        if np.all(int8_output == 0):
            print("Standard inference failed, trying alternative method...")
            int8_output = infer_with_latest_api(engine, context, dummy_input)
        
    end.record()
    torch.cuda.synchronize()


    elapsed_ms = start.elapsed_time(end)
    print(f"Inference time: {(elapsed_ms/20):.3f} ms")
    time.sleep(0.5)
    print(f"[nvidia-smi] 推理期间最大显存使用: {peak_mem_mb} MiB")