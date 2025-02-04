import onnx
import onnxruntime
import numpy as np
from pprint import pprint

def analyze_onnx_model(model_path):

    print("Load ONNX model...")
    model = onnx.load(model_path)

    print("\nModel metadata:")
    print(f"IR version: {model.ir_version}")
    print(f"Producer Name: {model.producer_name}")
    print(f"Producer Version: {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Model Version: {model.model_version}")

    print("\nModel Input:")
    for input in model.graph.input:
        print(f"\nInput name: {input.name}")
        print("Input shape:", end=" ")

        shape = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        print(shape)
        print(f"Data type: {input.type.tensor_type.elem_type}")

    print("\nModel Output:")
    for output in model.graph.output:
        print(f"\nOutput name: {output.name}")
        print("Output shape:", end=" ")
        shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        print(shape)
        print(f"Output data type: {output.type.tensor_type.elem_type}")
    
    print("\nNetwork structure:")
    for idx, node in enumerate(model.graph.node):
        print(f"\nNode {idx}:")
        print(f"Node op type: {node.op_type}")
        print(f"Inputs: {node.input}")
        print(f"Outputs: {node.output}")
        if node.attribute:
            print("Attributes:")
            for attr in node.attribute:
                print(f"  - {attr.name}: {attr}")

    print("\nRuntime information:")
    session = onnxruntime.InferenceSession(model_path)
    print("\nProviders:", session.get_providers())

    print("\nDetailed input information:")
    for i in session.get_inputs():
        print(f"name: {i.name}")
        print(f"shape: {i.shape}")
        print(f"type: {i.type}")
    
    print("\nDetailed output information:")
    for o in session.get_outputs():
        print(f"Name: {o.name}")
        print(f"Shape: {o.shape}")
        print(f"Type: {o.type}")

if __name__ == "__main__":
    model_path = "curve_model.onnx"
    analyze_onnx_model(model_path)
