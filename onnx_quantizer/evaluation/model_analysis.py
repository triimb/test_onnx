import os
from typing import Dict, Any, List
import onnx
import onnxruntime as ort
import numpy as np

def get_value_info_shape(model: onnx.ModelProto, value_name: str) -> List[int]:
    """
    Retrieve the shape of a tensor from the model's value_info, inputs, or outputs.
    
    Parameters:
        model (onnx.ModelProto): The ONNX model.
        value_name (str): Name of the tensor.
    
    Returns:
        List[int]: List of dimension values (using 1 as a fallback for unknown dims).
    """
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == value_name:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                # Use dim_value if set; otherwise, fallback to 1.
                dims.append(d.dim_value if d.dim_value > 0 else 1)
            return dims
    return []

def count_parameters(model_path: str) -> int:
    """
    Counts the number of learnable parameters in an ONNX model.
    
    Parameters:
        model_path (str): Path to the ONNX model.
    
    Returns:
        int: Total number of parameters.
    """
    model = onnx.load(model_path)
    total_params = 0
    for initializer in model.graph.initializer:
        total_params += np.prod(initializer.dims)
    return int(total_params)

def count_layers(model_path: str) -> int:
    """
    Counts the number of layers (i.e., nodes) in an ONNX model.
    
    Parameters:
        model_path (str): Path to the ONNX model.
    
    Returns:
        int: Total number of layers (nodes).
    """
    model = onnx.load(model_path)
    return len(model.graph.node)

def get_size(model_path: str) -> Dict[str, float]:
    """
    Gets the file size of the ONNX model in bytes, megabytes, and gigabytes.
    
    Parameters:
        model_path (str): Path to the ONNX model.
    
    Returns:
        dict: Dictionary with keys 'bytes', 'megabytes', and 'gigabytes'.
    """
    size_bytes = os.path.getsize(model_path)
    return {
        "bytes": size_bytes,
        "megabytes": size_bytes / (1024 * 1024),
        "gigabytes": size_bytes / (1024 * 1024 * 1024)
    }

def compute_gflops(model_path: str) -> float:
    """
    Estimates the number of GFLOPs required for a forward pass.
    
    The estimation is computed by summing the FLOPs of common operators (Conv and MatMul).
    For Conv, FLOPs = 2 * (M * (C/group) * kH * kW * outH * outW) where weight shape is [M, C, kH, kW].
    For MatMul, FLOPs = 2 * m * n * k, assuming input shapes (m, k) and (k, n).
    
    Parameters:
        model_path (str): Path to the ONNX model.
    
    Returns:
        float: Estimated GFLOPs (Giga FLOPs).
    """
    model = onnx.load(model_path)
    # Perform shape inference to get output shapes.
    model = onnx.shape_inference.infer_shapes(model)
    total_flops = 0.0

    for node in model.graph.node:
        if node.op_type == "Conv":
            if len(node.input) < 2:
                continue
            weight_name = node.input[1]
            weight = None
            for initializer in model.graph.initializer:
                if initializer.name == weight_name:
                    weight = initializer
                    break
            if weight is None:
                continue
            # Expect weight shape: [M, C/group, kH, kW]
            w_shape = list(weight.dims)
            if len(w_shape) < 4:
                continue
            M = w_shape[0]
            C = w_shape[1]
            kH = w_shape[2]
            kW = w_shape[3]
            group = 1
            for attr in node.attribute:
                if attr.name == "group":
                    group = attr.i
            # Retrieve the inferred output shape.
            if len(node.output) < 1:
                continue
            output_name = node.output[0]
            out_shape = get_value_info_shape(model, output_name)
            if len(out_shape) < 4:
                continue
            # Assume output shape: [N, M, outH, outW]
            outH = out_shape[2]
            outW = out_shape[3]
            conv_flops = 2 * M * (C // group) * kH * kW * outH * outW
            total_flops += conv_flops

        elif node.op_type == "MatMul":
            if len(node.input) < 2:
                continue
            shape_a = get_value_info_shape(model, node.input[0])
            shape_b = get_value_info_shape(model, node.input[1])
            if len(shape_a) < 2 or len(shape_b) < 2:
                continue
            # For matrix multiplication, assume shapes (m, k) and (k, n)
            m = shape_a[-2]
            k = shape_a[-1]
            n = shape_b[-1]
            matmul_flops = 2 * m * n * k
            total_flops += matmul_flops

    return total_flops / 1e9
