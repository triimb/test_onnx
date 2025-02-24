from typing import Dict, Any
from pathlib import Path

from onnxruntime.quantization.qdq_loss_debug import (
    modify_model_output_intermediate_tensors,
    collect_activations,
    create_activation_matching,
    compute_activation_error,
    create_weight_matching,
    compute_weight_error,
)
# Assume CalibrationDataReader is defined in the calibration module.
from calibrate import CalibrationDataReader

def compare_model_activations(
    float_model_path: str,
    qdq_model_path: str,
    augmented_model_path: str,
    data_reader: CalibrationDataReader,
    execution_providers: list = None
) -> Dict[str, Any]:
    """
    Compare activations between a float32 model and a quantized (QDQ) model.

    This wrapper performs the following steps:
      1. Augments the float model to include intermediate outputs by calling
         modify_model_output_intermediate_tensors.
      2. Collects activations from the augmented float model and from the QDQ model via collect_activations.
      3. Matches the corresponding activations using create_activation_matching and computes error metrics using
         compute_activation_error.
    
    Parameters:
        float_model_path (str): File path to the original float32 ONNX model.
        qdq_model_path (str): File path to the quantized (QDQ) ONNX model.
        augmented_model_path (str): File path where the augmented float model will be saved.
        data_reader (CalibrationDataReader): A data reader instance that supplies input data.
        execution_providers (list, optional): List of execution providers for ONNX Runtime.
            Defaults to ["CPUExecutionProvider"] if not provided.
        
    Returns:
        dict: A dictionary containing:
            - "activation_errors": Activation error metrics (in dB) for each matched tensor.
            - "float_model": The file name of the float model.
            - "qdq_model": The file name of the QDQ model.
            - "augmented_model": The file name of the augmented model.
    """
    # Augment the float model so that its intermediate tensors are saved.
    modify_model_output_intermediate_tensors(
        input_model_path=float_model_path,
        output_model_path=augmented_model_path
    )
    
    # Collect activations from the augmented float model.
    float_activations = collect_activations(
        augmented_model=augmented_model_path,
        input_reader=data_reader,
        execution_providers=execution_providers
    )
    
    # Collect activations from the quantized (QDQ) model.
    qdq_activations = collect_activations(
        augmented_model=qdq_model_path,
        input_reader=data_reader,
        execution_providers=execution_providers
    )
    
    # Create matching between activations and compute error metrics.
    matching = create_activation_matching(qdq_activations, float_activations)
    errors = compute_activation_error(matching)
    
    return {
        "activation_errors": errors,
        "float_model": Path(float_model_path).name,
        "qdq_model": Path(qdq_model_path).name,
        "augmented_model": Path(augmented_model_path).name,
    }


def compare_model_weights(
    float_model_path: str,
    qdq_model_path: str
) -> Dict[str, Any]:
    """
    Compare weights between a float32 model and a quantized (QDQ) model.

    This wrapper calls existing functions to match weights and compute an error metric (in dB)
    between the original weights and the dequantized weights from the QDQ model.
    
    Parameters:
        float_model_path (str): File path to the original float32 ONNX model.
        qdq_model_path (str): File path to the quantized (QDQ) ONNX model.
        
    Returns:
        dict: A dictionary containing:
            - "weight_errors": Weight error metrics (in dB) for each matched weight.
            - "float_model": The file name of the float model.
            - "qdq_model": The file name of the QDQ model.
    """
    matched_weights = create_weight_matching(float_model_path, qdq_model_path)
    errors = compute_weight_error(matched_weights)
    
    return {
        "weight_errors": errors,
        "float_model": Path(float_model_path).name,
        "qdq_model": Path(qdq_model_path).name,
    }
