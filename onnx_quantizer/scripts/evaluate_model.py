import argparse
import sys
from prettytable import PrettyTable
from evaluation.model_analysis import analyze_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate an ONNX model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU execution if available (affects inference-based shape inference).")
    args = parser.parse_args()

    try:
        # You can internally adjust analysis if GPU changes how shape inference is done.
        metrics = analyze_model(args.model)
    except Exception as e:
        print(f"Error evaluating model: {e}")
        sys.exit(1)

    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Model File", args.model])
    table.add_row(["Model Size (Bytes)", f'{metrics.get("model_size_bytes", 0):.2f}'])
    table.add_row(["Model Size (MB)", f'{metrics.get("model_size_MB", 0):.2f}'])
    table.add_row(["Model Size (GB)", f'{metrics.get("model_size_GB", 0):.2f}'])
    table.add_row(["# Parameters", metrics.get("num_parameters", "N/A")])
    table.add_row(["# Layers", metrics.get("num_layers", "N/A")])
    table.add_row(["Estimated GFLOPs", f'{metrics.get("GFLOPs", 0):.2f}'])
    
    # Display each input's name and shape.
    for inp in metrics.get("inputs", []):
        table.add_row([f"Input: {inp['name']}", f"Shape: {inp['shape']}"])
    
    print(table)

if __name__ == "__main__":
    main()
