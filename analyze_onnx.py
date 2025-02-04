import onnx
import onnxruntime
import numpy as np
from pprint import pprint

def analyze_onnx_model(model_path):
    # 1. 加载模型
    print("1. 加载ONNX模型...")
    model = onnx.load(model_path)
    
    # 2. 检查模型基本信息
    print("\n2. 模型基本信息:")
    print(f"IR版本: {model.ir_version}")
    print(f"生产者名称: {model.producer_name}")
    print(f"生产者版本: {model.producer_version}")
    print(f"域: {model.domain}")
    print(f"模型版本: {model.model_version}")
    
    # 3. 分析模型输入
    print("\n3. 模型输入:")
    for input in model.graph.input:
        print(f"\n输入名称: {input.name}")
        print("输入形状:", end=" ")
        # 获取输入张量的形状信息
        shape = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        print(shape)
        print(f"数据类型: {input.type.tensor_type.elem_type}")
    
    # 4. 分析模型输出
    print("\n4. 模型输出:")
    for output in model.graph.output:
        print(f"\n输出名称: {output.name}")
        print("输出形状:", end=" ")
        shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        print(shape)
        print(f"数据类型: {output.type.tensor_type.elem_type}")
    
    # 5. 分析网络结构
    print("\n5. 网络结构:")
    for idx, node in enumerate(model.graph.node):
        print(f"\n节点 {idx}:")
        print(f"操作类型: {node.op_type}")
        print(f"输入: {node.input}")
        print(f"输出: {node.output}")
        if node.attribute:
            print("属性:")
            for attr in node.attribute:
                print(f"  - {attr.name}: {attr}")

    # 6. 获取运行时信息
    print("\n6. 运行时信息:")
    session = onnxruntime.InferenceSession(model_path)
    print("\n可用的执行提供程序:", session.get_providers())
    
    # 打印输入输出详细信息
    print("\n模型输入详细信息:")
    for i in session.get_inputs():
        print(f"名称: {i.name}")
        print(f"形状: {i.shape}")
        print(f"类型: {i.type}")
    
    print("\n模型输出详细信息:")
    for o in session.get_outputs():
        print(f"名称: {o.name}")
        print(f"形状: {o.shape}")
        print(f"类型: {o.type}")

if __name__ == "__main__":
    model_path = "curve_model.onnx"
    analyze_onnx_model(model_path)
