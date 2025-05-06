import torch
import numpy as np
import argparse

from safetensors.torch import load_file

import torch.onnx

from TimerXL.models.timer_xl import Model
from TimerXL.models.configuration_timer import TimerxlConfig

class ONNXExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="input path")
    parser.add_argument("--output_path", required=True, help="output path")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    # Example usage
    args = get_args()
    model = Model(TimerxlConfig())
    state_dict = load_file(args.input_path)
    state_dict = {'model.' + k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model.to(device=device)
    model.eval()
    
    model = ONNXExportWrapper(model)
    
    example_input = torch.randn(1, 672).to(device)
    
    torch.onnx.export(
        model,
        example_input,
        args.output_path,
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )
    
    import onnx
    import numpy as np
    import onnxruntime as ort
    model_onnx_path = './model.onnx'
    # 验证模型的合法性
    onnx_model = onnx.load(model_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型的输入名称:")
    for input in onnx_model.graph.input:
        print(f"- {input.name}")
    
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 1  # 设置更详细的日志级别
    
    # 创建ONNX运行时会话
    ort_session = ort.InferenceSession(
        model_onnx_path, 
        sess_options=sess_options,
        # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        providers=['CUDAExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    print(f"使用输入名称: {input_name}")
    # 准备输入数据
    input_data = {
        input_name: example_input.cpu().numpy()
    }
    
    # 运行推理
    for i in range(3):
        y_pred_onnx = ort_session.run(None, input_data)
        
        
    import pandas as pd
    df = pd.read_csv('/data/mahaoke/AINode/ainode/TimerXL/data/ETT-small/ETTh2.csv')
        
    seq_len = 672
    pred_len= 96
    point=672

    input_data = df["OT"][point-seq_len:point].values
    # 转float32
    input_data = input_data.astype(np.float32)
    # # 现在的形状是[672]，转为[16,672]（复制16份）
    input_data = np.tile(input_data, (1, 1))
    input_data = {
        input_name: input_data
    }
        
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    torch.cuda.synchronize()
    
    y_pred_onnx = ort_session.run(None, input_data)
    
    ender.record()
    torch.cuda.synchronize()
    elapsed_time = starter.elapsed_time(ender)
    print(f"Elapsed time: {elapsed_time} ms")