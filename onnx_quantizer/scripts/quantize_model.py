#!/usr/bin/env python3
"""
quantize_model.py

Quantize an ONNX model using a YAML configuration file for parameters.
The configuration file (default: configs/quantization.yaml) should include model paths,
quantization settings (precision, method, etc.), calibration parameters (if needed),
and a list of execution providers.

Usage:
    python quantize_model.py [--config configs/quantization.yaml] [--gpu]
"""

import argparse
import sys
from pathlib import Path
import yaml
from prettytable import PrettyTable

# Import the quantization wrapper from the backends.
from backends.onnx_quantization import quantize_model

# Optional: Import a helper function to create a calibration data reader.
try:
    from calibration.data_loader import create_calibration_data_reader
except ImportError:
    # If not available, define a dummy function.
    def create_calibration_data_reader(config):
        raise NotImplementedError("Calibration data reader creation is not implemented.")


def load_config(config_path: str) -> dict:
    """Load the YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration file {config_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Quantize an ONNX model using configuration settings.")
    parser.add_argument("--config", type=str, default="configs/quantization.yaml", help="Path to the YAML config file.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU execution provider if available.")
    args = parser.parse_args()

    config = load_config(args.config)

    # Extract model file paths.
    try:
        model_config = config.get("model", {})
        input_model_path = model_config["input"]
        output_model_path = model_config["output"]
    except KeyError as e:
        print(f"Missing model configuration key: {e}")
        sys.exit(1)

    # Extract quantization settings.
    quant_config = config.get("quantization", {})
    precision = quant_config.get("precision", "int8")
    quantization_type = quant_config.get("method", "static")
    calibrate_method = quant_config.get("calibrate_method", "MinMax")
    quant_format = quant_config.get("quant_format", "QOperator")
    activation_type = quant_config.get("activation_type", "QUInt8")
    weight_type = quant_config.get("weight_type", "QInt8")
    per_channel = quant_config.get("per_channel", True)
    reduce_range = quant_config.get("reduce_range", False)

    # Determine execution providers.
    providers = config.get("providers", ["CPUExecutionProvider"])
    # Override providers if --gpu flag is passed.
    if args.gpu:
        # If GPU is requested, pick the first GPU provider from config, or force CUDA if not present.
        providers = [p for p in providers if "CUDA" in p] or ["CUDAExecutionProvider"]

    # Prepare extra keyword arguments for quantization.
    extra_kwargs = {
        "calibrate_method": calibrate_method,
        "quant_format": quant_format,
        "activation_type": activation_type,
        "weight_type": weight_type,
        "per_channel": per_channel,
        "reduce_range": reduce_range,
    }

    # For static INT8 quantization, a calibration data reader is required.
    if precision.lower() == "int8" and quantization_type.lower() == "static":
        calib_config = config.get("calibration", {})
        try:
            calibration_data_reader = create_calibration_data_reader(calib_config)
            extra_kwargs["calibration_data_reader"] = calibration_data_reader
        except Exception as e:
            print(f"Error creating calibration data reader: {e}")
            sys.exit(1)

    # Now call the quantization process.
    try:
        quantized_model = quantize_model(
            model_path=input_model_path,
            output_path=output_model_path,
            precision=precision,
            quantization_type=quantization_type,
            providers=providers,
            **extra_kwargs
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during quantization: {e}")
        sys.exit(1)

    # Display results
    table = PrettyTable()
    table.field_names = ["Parameter", "Value"]
    table.add_row(["Input Model", input_model_path])
    table.add_row(["Output Model", output_model_path])
    table.add_row(["Precision", precision])
    table.add_row(["Quantization Type", quantization_type])
    table.add_row(["Calibrate Method", calibrate_method])
    table.add_row(["Quant Format", quant_format])
    table.add_row(["Activation Type", activation_type])
    table.add_row(["Weight Type", weight_type])
    table.add_row(["Per Channel", per_channel])
    table.add_row(["Reduce Range", reduce_range])
    table.add_row(["Execution Provider", providers[0]])
    
    print("Quantization completed successfully. Summary:")
    print(table)


if __name__ == "__main__":
    main()
