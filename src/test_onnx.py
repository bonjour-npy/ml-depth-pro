import onnx

# 加载模型
model_path = "./onnx_weights/model_fp16.onnx"
model = onnx.load(model_path)

# 验证模型
onnx.checker.check_model(model)

# 获取输入数据类型
for input_tensor in model.graph.input:
    input_name = input_tensor.name
    data_type = input_tensor.type.tensor_type.elem_type
    print(
        f"Input Name: {input_name}, Data Type: {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[data_type]}"
    )

# 打印输入节点的名称、类型和形状
print("Model Inputs:")
for input_tensor in model.graph.input:
    input_name = input_tensor.name
    input_type = input_tensor.type.tensor_type.elem_type  # 数据类型
    input_shape = [
        dim.dim_value if dim.dim_value > 0 else "?"
        for dim in input_tensor.type.tensor_type.shape.dim
    ]  # 获取形状
    print(f"Input Name: {input_name}, Type: {input_type}, Shape: {input_shape}")

# 打印输出节点的名称、类型和形状
print("\nModel Outputs:")
for output_tensor in model.graph.output:
    output_name = output_tensor.name
    output_type = output_tensor.type.tensor_type.elem_type  # 数据类型
    output_shape = [
        dim.dim_value if dim.dim_value > 0 else "?"
        for dim in output_tensor.type.tensor_type.shape.dim
    ]  # 获取形状
    print(f"Output Name: {output_name}, Type: {output_type}, Shape: {output_shape}")
