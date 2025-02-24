from onnx import ModelProto
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_dynamic_model(
    model_path: str,
    output_path: str,
    op_types_to_quantize=None,
    per_channel: bool = False,
    reduce_range: bool = False,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    use_external_data_format: bool = False,
    extra_options: dict = None,
) -> ModelProto:
    """
    Apply dynamic INT8 quantization to an ONNX model using onnxruntime's quantization tools.

    Dynamic quantization does not require calibration data; the activation ranges are computed on-the-fly.
    This minimal wrapper calls onnxruntime.quantization.quantize_dynamic and integrates execution providers
    via extra_options.

    Parameters:
        model_path (str):
            Path to the original ONNX model.
        
        output_path (str):
            Path to save the quantized model.
        
        op_types_to_quantize (list, optional):
            List of operator types to quantize (e.g., ["Conv", "MatMul"]).
        
        per_channel (bool, default=False):
            If True, applies per-channel quantization.
        
        reduce_range (bool, default=False):
            If True, uses a reduced quantization range.
        
        weight_type:
            Data type for weights (e.g., QuantType.QInt8 or QuantType.QUInt8).
        
        nodes_to_quantize (list, optional):
            List of specific node names to quantize.
        
        nodes_to_exclude (list, optional):
            List of node names to exclude from quantization.
        
        use_external_data_format (bool, default=False):
            If True, saves the quantized model using an external data format.
        
        extra_options (dict, optional):
            Additional options for fine-tuning quantization.
        
    Returns:
        ModelProto: The dynamically quantized ONNX model.
    """
    return quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        op_types_to_quantize=op_types_to_quantize,
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=weight_type,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        use_external_data_format=use_external_data_format,
        extra_options=extra_options
    )
