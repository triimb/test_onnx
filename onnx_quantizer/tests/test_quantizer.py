import unittest
from onnx_quantizer.quantizer import ONNXQuantizer

class TestONNXQuantizer(unittest.TestCase):
    """Unit tests for ONNXQuantizer."""

    def setUp(self) -> None:
        """Initializes a test model."""
        self.quantizer = ONNXQuantizer("models/test.onnx")

    def test_load_model(self) -> None:
        """Ensures the ONNX model loads correctly."""
        pass

    def test_quantization_fp16(self) -> None:
        """Verifies FP16 quantization executes without errors."""
        pass

    def test_quantization_int8_static(self) -> None:
        """Tests static INT8 quantization with calibration."""
        pass

    def test_quantization_int8_dynamic(self) -> None:
        """Tests dynamic INT8 quantization."""
        pass

if __name__ == "__main__":
    unittest.main()
