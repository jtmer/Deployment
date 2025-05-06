from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import onnxruntime as ort
import onnx
import numpy as np
import pandas as pd
import argparse

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name, raw_series: np.ndarray, window_size: int = 672, max_samples: int = 100):
        self.input_name = input_name
        self.window_size = window_size
        self.samples = []
        
        total = len(raw_series) - window_size + 1
        num_samples = min(max_samples, total)
        for i in range(num_samples):
            x = raw_series[i:i+window_size].reshape(1, -1).astype(np.float32)
            self.samples.append({input_name: x})
        
        self.enum_data = iter(self.samples)

    def get_next(self):
        return next(self.enum_data, None)

def run_onnx_model(session, input_name, input_tensor):
    return session.run(None, {input_name: input_tensor})[0]

def evaluate_models(fp32_path, int8_path, data, seq_len=672, pred_len=96):
    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name

    total_windows = len(data) - seq_len - pred_len + 1
    print(f"共有 {total_windows} 个滑动窗口")

    mses, maes = [], []

    for i in range(total_windows):
        x = data[i:i+seq_len].reshape(1, -1).astype(np.float32)
        y_true = data[i+seq_len:i+seq_len+pred_len]

        y_fp32 = run_onnx_model(sess_fp32, input_name, x).ravel()
        y_int8 = run_onnx_model(sess_int8, input_name, x).ravel()

        # 保留前 pred_len 个元素
        y_fp32 = y_fp32[:pred_len]
        y_int8 = y_int8[:pred_len]

        mse = np.mean((y_int8 - y_fp32) ** 2)
        mae = np.mean(np.abs(y_int8 - y_fp32))

        mses.append(mse)
        maes.append(mae)

        if i % 100 == 0 or i == total_windows - 1:
            print(f"[{i+1}/{total_windows}] MSE={mse:.4f}, MAE={mae:.4f}")

    print("\n====== INT8 评估结果 ======")
    print(f"窗口总数: {len(mses)}")
    print(f"平均 MSE: {np.mean(mses):.6f}")
    print(f"平均 MAE: {np.mean(maes):.6f}")
    print(f"最大 MSE: {np.max(mses):.6f}")
    print(f"最大 MAE: {np.max(maes):.6f}")
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="input path")
    parser.add_argument("--output_path", required=True, help="output path")
    parser.add_argument(
        "--calibrate_dataset", default="", help="calibration data set"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
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
    
    df_first = pd.read_csv(file_path, nrows=1000)
    first_data = df_first["OT"].values.astype(np.float32)
    print(f"读取了前 {len(first_data)} 行数据")

    middle_start = total_rows // 2 - 500
    middle_end = middle_start + 1000
    df_middle = pd.read_csv(file_path, skiprows=range(1, middle_start + 1), nrows=1000)
    middle_data = df_middle["OT"].values.astype(np.float32)
    print(f"读取了中间 {len(middle_data)} 行数据 (从第 {middle_start} 行开始)")
    
    df_last = pd.read_csv(file_path, skiprows=range(1, total_rows - 1000))
    last_data = df_last["OT"].values.astype(np.float32)
    print(f"读取了最后 {len(last_data)} 行数据")

    # 合并所有数据用于评估
    all_data = np.concatenate([first_data, middle_data, last_data])
    print(f"合并后的数据总量: {len(all_data)}")
    
    onnx_model = onnx.load(args.input_path)
    input_name = onnx_model.graph.input[0].name

    calibration_dataset = MyCalibrationDataReader(input_name, all_data)

    quantize_static(
        model_input=args.input_path,
        model_output=args.output_path,
        calibration_data_reader=calibration_dataset,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )

    print(f"INT8 静态量化完成")
    
    evaluate_models(args.input_path, args.output_path, all_data, 672, 96)
