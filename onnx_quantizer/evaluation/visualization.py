import netron

def launch_netron(model_path: str, port: int = 8080) -> None:
    """
    Launches Netron to visualize the specified ONNX model.
    
    Parameters:
        model_path (str): Path to the ONNX model file.
        port (int, default=8080): Port on which to run the Netron server.
    """
    netron.start(model_path, port=port)
