import pandas as pd
import numpy as np
    
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

import argparse

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
            fixed_shape[0] = 1  # 使用足够大的批量大小，根据实际需求调整
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

# ------------------------ Calibration Loaders ------------------------
class TimeSeriesCalibLoader:
    def __init__(self, ts_data: np.ndarray, batch_size: int, calib_count: int):
        self.data = ts_data.astype(np.float32)
        self.batch_size = batch_size
        self.calib_count = calib_count
        self.index = 0
        self.total = min(len(ts_data), batch_size * calib_count)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index * self.batch_size < self.total:
            start = self.index * self.batch_size
            end = start + self.batch_size
            batch = self.data[start:end]
            if batch.shape[0] < self.batch_size:
                pad = np.zeros((self.batch_size - batch.shape[0], batch.shape[1]), dtype=np.float32)
                batch = np.concatenate([batch, pad], axis=0)
            # batch = batch[:, np.newaxis, :]  # [B, 1, L]
            self.index += 1
            return np.ascontiguousarray(batch)
        else:
            return np.array([])

# ------------------------ Calibrators ------------------------
class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, dataloader: TimeSeriesCalibLoader, cache_file="minmax_calib.cache"):
        super().__init__()
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.device_input = cuda.mem_alloc(self.dataloader.data.nbytes // self.dataloader.calib_count)
        self.dataloader.reset()

    def get_batch_size(self):
        return self.dataloader.batch_size

    def get_batch(self, names):
        batch = self.dataloader.next_batch()
        if batch.size == 0:
            return None
        cuda.memcpy_htod(self.device_input, batch)
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader: TimeSeriesCalibLoader, cache_file="entropy_calib.cache"):
        super().__init__()
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.device_input = cuda.mem_alloc(self.dataloader.data.nbytes // self.dataloader.calib_count)
        self.dataloader.reset()

    def get_batch_size(self):
        return self.dataloader.batch_size

    def get_batch(self, names):
        batch = self.dataloader.next_batch()
        if batch.size == 0:
            return None
        cuda.memcpy_htod(self.device_input, batch)
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Placeholder: implement percentile-based quantization calibration here
class PercentileCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, dataloader: TimeSeriesCalibLoader, percentile=99.9, cache_file="percentile_calib.cache"):
        super().__init__()
        self.dataloader = dataloader
        self.percentile = percentile
        self.cache_file = cache_file
        self.device_input = cuda.mem_alloc(self.dataloader.data.nbytes // self.dataloader.calib_count)
        self.dataloader.reset()

    def get_batch_size(self):
        return self.dataloader.batch_size

    def get_batch(self, names):
        batch = self.dataloader.next_batch()
        if batch.size == 0:
            return None
        # NOTE: in real implementation you would clip based on percentile here before copying
        cuda.memcpy_htod(self.device_input, batch)
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ------------------------ Build Engine ------------------------
def build_int8_engine(onnx_path: str, engine_path: str, dataloader: TimeSeriesCalibLoader, method="entropy", use_fp16: bool = False):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model.")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    
    # 添加优化配置文件
    print("Adding optimization profile for dynamic inputs...")
    profile = builder.create_optimization_profile()
    
    # 获取输入名称和形状
    input_name = network.get_input(0).name
    print(f"Model input name: {input_name}")
    
    # 为动态批大小设置最小、优化和最大形状
    min_batch = 1
    opt_batch = dataloader.batch_size
    max_batch = dataloader.batch_size * 2  # 设置一个足够大的最大批大小
    
    # 根据输入维度设置配置文件
    if network.get_input(0).shape[0] == -1:  # 检查批大小是否为动态
        input_shape = network.get_input(0).shape
        if len(input_shape) == 2:  # [batch_size, seq_len]
            seq_len = input_shape[1] if input_shape[1] != -1 else 672
            profile.set_shape(input_name, (min_batch, seq_len), (opt_batch, seq_len), (max_batch, seq_len))
            print(f"Set profile for 2D input: min=({min_batch}, {seq_len}), opt=({opt_batch}, {seq_len}), max=({max_batch}, {seq_len})")
        elif len(input_shape) == 3:  # [batch_size, channels, seq_len]
            channels = input_shape[1] if input_shape[1] != -1 else 1
            seq_len = input_shape[2] if input_shape[2] != -1 else 672
            profile.set_shape(input_name, (min_batch, channels, seq_len), (opt_batch, channels, seq_len), (max_batch, channels, seq_len))
            print(f"Set profile for 3D input: min=({min_batch}, {channels}, {seq_len}), opt=({opt_batch}, {channels}, {seq_len}), max=({max_batch}, {channels}, {seq_len})")
    
    config.add_optimization_profile(profile)

    if method == "minmax":
        config.int8_calibrator = MinMaxCalibrator(dataloader)
    elif method == "entropy":
        config.int8_calibrator = EntropyCalibrator(dataloader)
    elif method == "percentile":
        config.int8_calibrator = PercentileCalibrator(dataloader, percentile=99.9)
    else:
        raise ValueError("Unsupported calibration method: " + method)

    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Building TensorRT INT8 engine with {method} calibration...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine:
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        print(f"INT8 engine saved to: {engine_path}")
        
        # 创建运行时和引擎
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine
    else:
        print("Failed to build the engine.")
        return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="input path")
    parser.add_argument("--output_path", required=True, help="output path")
    parser.add_argument(
        "--calibrate_dataset", default="/data/mahaoke/AINode/ainode/TimerXL/data/ETT-small/ETTh2.csv", help="calibration data set"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    input_data = pd.read_csv(args.calibrate_dataset)['OT'].values
    input_data = input_data.astype(np.float32)
    loader = TimeSeriesCalibLoader(input_data, batch_size=672, calib_count=20)
    build_int8_engine(args.input_path, args.output_path, loader, method="entropy", use_fp16=False)
    
    print("验证误差...")
    file_path = args.calibrate_dataset
    try:
        import subprocess
        result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
        total_rows = int(result.stdout.split()[0])
        print(f"估计的总行数: {total_rows}")
    except Exception as e:
        print(f"无法使用wc命令: {e}")
        # 方法2：只读取文件行数（效率较低但兼容性好）
        with open(file_path, 'r') as f:
            total_rows = sum(1 for _ in f)
        print(f"计算的总行数: {total_rows}")
    
    # 读取前1万行
    df_first = pd.read_csv(file_path, nrows=1000)
    first_data = df_first["OT"].values.astype(np.float32)
    print(f"读取了前 {len(first_data)} 行数据")

    # 读取中间1万行
    middle_start = total_rows // 2 - 500
    middle_end = middle_start + 1000
    df_middle = pd.read_csv(file_path, skiprows=range(1, middle_start + 1), nrows=1000)
    middle_data = df_middle["OT"].values.astype(np.float32)
    print(f"读取了中间 {len(middle_data)} 行数据 (从第 {middle_start} 行开始)")

    # 读取最后1万行
    df_last = pd.read_csv(file_path, skiprows=range(1, total_rows - 1000))
    last_data = df_last["OT"].values.astype(np.float32)
    print(f"读取了最后 {len(last_data)} 行数据")

    all_data = np.concatenate([first_data, middle_data, last_data])
    
    engine = load_engine(args.output_path)
    context = engine.create_execution_context()
    
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    # 批量预测参数
    seq_len = 672  # 输入长度
    pred_len = 96  # 预测长度(应与输出tensor形状匹配)
    
    # 记录所有预测结果和统计信息
    all_predictions = []
    all_ground_truth = []
    all_mse = []
    all_mae = []
    
    # 创建ONNX会话用于FP32参考
    import onnxruntime as ort
    onnx_path = "model.onnx"
    onnx_session = ort.InferenceSession(onnx_path)
    input_name = onnx_session.get_inputs()[0].name
    
    total_samples = len(all_data)
    # 滑动窗口预测
    for i in range(0, total_samples - seq_len - pred_len + 1):
        # 准备输入数据
        input_window = all_data[i:i+seq_len]
        input_tensor = input_window.reshape(1, -1)  # [1, seq_len]
        
        # INT8 推理
        print(f"\nProcessing window {i+1}/{total_samples - seq_len - pred_len + 1}...")
        try:
            # 使用标准方法
            int8_output = infer(context, bindings, inputs, outputs, stream, input_tensor)
            
            # 如果输出全为0，使用备用方法
            if np.all(int8_output == 0):
                print("Standard inference failed, trying alternative method...")
                int8_output = infer_with_latest_api(engine, context, input_tensor)
            
            # FP32参考
            fp32_output = onnx_session.run(None, {input_name: input_tensor})[0].ravel()
            
            # 确保输出形状合理
            int8_pred = int8_output[:pred_len] if int8_output.size >= pred_len else int8_output
            fp32_pred = fp32_output[:pred_len] if fp32_output.size >= pred_len else fp32_output
            
            # 计算评估指标
            mse = np.mean((int8_pred - fp32_pred) ** 2)
            mae = np.mean(np.abs(int8_pred - fp32_pred))
            
            # 保存结果
            all_predictions.append(int8_pred)
            all_ground_truth.append(fp32_pred)
            all_mse.append(mse)
            all_mae.append(mae)
            
            print(f"Window {i+1} MSE: {mse:.4f}, MAE: {mae:.4f}")
            
        except Exception as e:
            print(f"Error processing window {i+1}: {e}")
            continue
    
    # 计算整体评估结果
    max_mse_idx = np.argmax(all_mse)
    max_mae_idx = np.argmax(all_mae)
    
    print("\n========== INT8 评估结果 ==========")
    print(f"总评估窗口数量: {len(all_mse)}")
    print(f"平均 MSE: {np.mean(all_mse):.4f}")
    print(f"平均 MAE: {np.mean(all_mae):.4f}")
    print(f"最大 MSE: {np.max(all_mse):.4f} (窗口索引 {max_mse_idx+1})")
    print(f"最大 MAE: {np.max(all_mae):.4f} (窗口索引 {max_mae_idx+1})")