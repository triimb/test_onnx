from onnx import ModelProto
from onnxconverter_common.float16 import convert_float_to_float16

def quantize_fp16_model(
    model_path: str,
    min_positive_val: float = 1e-7,
    max_finite_val: float = 1e4,
    keep_io_types: bool = False,
    disable_shape_infer: bool = False,
    op_block_list: list = None,
    node_block_list: list = None,
    check_fp16_ready: bool = True,
) -> ModelProto:
    """
    Convert an ONNX model to FP16 precision using Xiaowu's new implementation.

    This minimal wrapper loads an FP32 ONNX model from the given path, converts it to FP16 using the
    `convert_float_to_float16` function (which fixes several bugs from prior ORT changes), returns the FP16 model.

    Parameters:
        model_path (str):
            Path to the original FP32 ONNX model.
        
        min_positive_val (float, default=1e-7):
            Minimum positive value threshold for conversion. Values below this threshold will be adjusted.
        
        max_finite_val (float, default=1e4):
            Maximum finite value threshold for conversion. Values above this threshold will be adjusted.
        
        keep_io_types (bool, default=False):
            If True, retains the input and output data types in FP32 while converting internal weights to FP16.
        
        disable_shape_infer (bool, default=False):
            If True, disables shape inference during conversion.
        
        op_block_list (list, optional):
            List of operator types to exclude from FP16 conversion.
        
        node_block_list (list, optional):
            List of specific node names to exclude from FP16 conversion.
        
        check_fp16_ready (bool, default=True):
            If True, checks whether the model is ready for FP16 conversion and raises an error if not.
        
    Returns:
        ModelProto:
            The ONNX model converted to FP16 precision.
    """

    # Load the original model.
    model = onnx.load(model_path)

    # Convert the model to FP16 using Xiaowu's new implementation.
    return convert_float_to_float16(
        model,
        min_positive_val=min_positive_val,
        max_finite_val=max_finite_val,
        keep_io_types=keep_io_types,
        disable_shape_infer=disable_shape_infer,
        op_block_list=op_block_list,
        node_block_list=node_block_list,
        check_fp16_ready=check_fp16_ready
    )

