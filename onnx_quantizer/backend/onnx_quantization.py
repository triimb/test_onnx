from onnx import ModelProto
from .quantize_static import quantize_static_model
from .quantize_dynamic import quantize_dynamic_model
from .quantize_fp16 import quantize_fp16_model

def quantize_model(
    model_path: str,
    precision: str,
    quantization_type: str,
    providers: list = None,
    **kwargs
) -> ModelProto:
    """
    Dispatch the quantization process based on the desired precision and quantization type,
    using the ONNX Runtime quantization functions. This is a minimal wrapper that passes the
    model path and output path, along with all additional parameters, to the underlying quantization
    methods.

    Parameters:
        model_path (str):
            Path to the original ONNX model.
        
        precision (str):
            The target precision for quantization. Supported values:
              - "fp16": Convert the model to FP16.
              - "int8": Quantize the model to INT8.
        
        quantization_type (str):
            The quantization method to use. Supported values depend on the precision:
              - For "fp16": Only "dynamic" is supported (static quantization is not applicable for FP16 conversion).
              - For "int8": Both "static" and "dynamic" methods are supported.
        
        providers (list, optional):
            List of execution providers (e.g., ["CUDAExecutionProvider"] or ["CPUExecutionProvider"]).
            Defaults to ["CPUExecutionProvider"] if not provided.
        
        **kwargs:
            Additional parameters to pass to the underlying quantization function. These should match the
            requirements of the specific quantization method being applied.

    Returns:
        ModelProto:
            The quantized ONNX model.

    Raises:
        ValueError: If an unsupported precision or quantization type is provided.
    """
    precision = precision.lower()
    quantization_type = quantization_type.lower()

    if precision == 'fp16':
        if quantization_type != 'dynamic':
            raise ValueError(
                "For FP16 conversion, only dynamic conversion is supported. Received quantization_type: '{}'."
                .format(quantization_type)
            )
        return quantize_fp16_model(model_path, providers=providers, **kwargs)
    elif precision == 'int8':
        if quantization_type == 'static':
            return quantize_static_model(model_path, providers=providers, **kwargs)
        elif quantization_type == 'dynamic':
            return quantize_dynamic_model(model_path, providers=providers, **kwargs)
        else:
            raise ValueError("Unsupported quantization type for INT8: '{}'.".format(quantization_type))
    else:
        raise ValueError("Unsupported precision: '{}'. Supported precisions are 'fp16' and 'int8'.".format(precision))
