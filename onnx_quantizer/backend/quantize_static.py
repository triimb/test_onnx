from onnx import ModelProto
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

def quantize_static_model(
    model_path: str,
    output_path: str,
    calibration_data_reader: CalibrationDataReader,
    calibrate_method: str = "MinMax",
    quant_format: str = "QOperator",
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=None,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    per_channel: bool = False,
    reduce_range: bool = False,
    use_external_data_format: bool = False,
    providers: list = None,
    extra_options: dict = None
) -> ModelProto:
    """
    Apply static INT8 quantization to an ONNX model using onnxruntime's quantization tools.

    This minimal wrapper calls onnxruntime.quantization.quantize_static, integrating the execution
    providers into the extra_options. Static quantization requires a calibration dataset (provided by
    calibration_data_reader) to compute activation ranges before quantization.

    Parameters:
        model_path (str):
            Path to the original ONNX model.
        
        output_path (str):
            Path to save the quantized model.

        calibration_data_reader (CalibrationDataReader):
            A data reader instance that provides input tensors for calibration.
        
        calibrate_method (str, default="MinMax"):
            The method to compute activation ranges. Options include:
              - "MinMax": Uses min/max values.
              - "Entropy": Uses an entropy-based approach.
        
        quant_format (str, default="QOperator"):
            Quantization format. Options:
              - "QOperator": Uses quantized operators.
              - "QDQ": Uses Quantize-Dequantize nodes.
        
        activation_type:
            Data type for activations (e.g., QuantType.QUInt8 or QuantType.QInt8).
        
        weight_type:
            Data type for weights (e.g., QuantType.QInt8 or QuantType.QUInt8).
        
        op_types_to_quantize (list, optional):
            List of operator types to quantize (e.g., ["Conv", "MatMul"]).
        
        nodes_to_quantize (list, optional):
            List of specific node names to quantize.
        
        nodes_to_exclude (list, optional):
            List of node names to exclude from quantization.
        
        per_channel (bool, default=False):
            If True, applies per-channel quantization (beneficial for convolution layers).
        
        reduce_range (bool, default=False):
            If True, uses a reduced quantization range (7-bit instead of 8-bit).
        
        use_external_data_format (bool, default=False):
            If True, saves the quantized model using an external data format (useful for models >2GB).
        
        extra_options (dict, optional):
            Additional options for fine-tuning quantization.
        
        providers (list, optional):
            List of execution providers to use (e.g., ["CUDAExecutionProvider"] or ["CPUExecutionProvider"]).
            Defaults to ["CPUExecutionProvider"] if not provided.

    Returns:
        ModelProto: The statically quantized ONNX model.
    """

    return quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_data_reader,
        calibrate_method=calibrate_method,
        quant_format=QuantFormat[quant_format],
        activation_type=activation_type,
        weight_type=weight_type,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=per_channel,
        reduce_range=reduce_range,
        use_external_data_format=use_external_data_format,
        calibration_providers=providers,
        extra_options=extra_options
    )
