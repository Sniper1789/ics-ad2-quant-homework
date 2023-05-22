from onnxruntime.quantization import quantize_dynamic,QuantType,CalibrationDataReader,QuantFormat,quantize_static,CalibrationMethod
import onnxruntime
import numpy as np
import time
def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 224, 224), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")

onnx_model = "./resnet50-v2-7.onnx"
onnx_model2 = "./resnet50-v2-7sim.onnx"
# model_quant_dynamic = "mnist-12_dynamic.onnx"
# quantize_dynamic(onnx_model,model_quant_dynamic,weight_type=QuantType.QUInt8)
# print("benchmarking fp32 model...")
# benchmark(onnx_model)
#
# print("benchmarking int8 model...")
# benchmark(model_quant_dynamic)
import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load(onnx_model)

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"


onnx.save(model_simp,onnx_model2)

# use model_simp as a standard ONNX model object