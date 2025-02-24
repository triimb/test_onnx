import argparse
import sys
from prettytable import PrettyTable
from model_comparison import compare_model_activations, compare_model_weights

# For demonstration, we define a simple dummy data reader.
import numpy as np
from calibrate import CalibrationDataReader

class DummyDataReader(CalibrationDataReader):
    """
    Dummy implementation of CalibrationDataReader that yields random inputs.
    In practice, replace this with your actual data loading logic.
    """
    def __init__(self, input_name="input", num_batches=5, batch_size=1):
        self.input_name = input_name
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.current = 0

    def get_next(self):
        if self.current < self.num_batches:
            self.current += 1
            return {self.input_name: np.random.rand(self.batch_size, 3, 224, 224).astype(np.float32)}
        return None

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        data = self.get_next()
        if data is None:
            raise StopIteration
        return data

def main():
    parser = argparse.ArgumentParser(description="Compare float and quantized ONNX models.")
    parser.add_argument("--float_model", type=str, required=True, help="Path to the float ONNX model.")
    parser.add_argument("--qdq_model", type=str, required=True, help="Path to the quantized (QDQ) ONNX model.")
    parser.add_argument("--augmented_model", type=str, required=True, help="Path to save the augmented float model.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU execution provider if available.")
    args = parser.parse_args()

    # Create a dummy data reader (replace with your real data reader if available).
    data_reader = DummyDataReader(input_name="input", num_batches=5, batch_size=1)
    providers = ["CUDAExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]

    try:
        act_results = compare_model_activations(
            float_model_path=args.float_model,
            qdq_model_path=args.qdq_model,
            augmented_model_path=args.augmented_model,
            data_reader=data_reader,
            execution_providers=providers
        )
    except Exception as e:
        print(f"Error comparing activations: {e}")
        sys.exit(1)

    try:
        weight_results = compare_model_weights(
            float_model_path=args.float_model,
            qdq_model_path=args.qdq_model
        )
    except Exception as e:
        print(f"Error comparing weights: {e}")
        sys.exit(1)

    # Display activation errors in a table.
    act_table = PrettyTable()
    act_table.field_names = ["Tensor Name", "QDQ Error (dB)", "Float vs QDQ Error (dB)"]
    for tensor, errors in act_results["activation_errors"].items():
        qdq_err = errors.get("qdq_err", "N/A")
        xmodel_err = errors.get("xmodel_err", "N/A")
        act_table.add_row([tensor, f"{qdq_err:.2f}" if isinstance(qdq_err, (int, float)) else qdq_err,
                           f"{xmodel_err:.2f}" if isinstance(xmodel_err, (int, float)) else xmodel_err])
    print("Activation Comparison:")
    print(act_table)

    # Display weight errors in a table.
    weight_table = PrettyTable()
    weight_table.field_names = ["Weight Name", "Error (dB)"]
    for weight, err in weight_results["weight_errors"].items():
        weight_table.add_row([weight, f"{err:.2f}" if isinstance(err, (int, float)) else err])
    print("\nWeight Comparison:")
    print(weight_table)

if __name__ == "__main__":
    main()
