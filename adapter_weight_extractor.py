from safetensors.torch import load_file
import torch
import sys

def extract_weights(safetensor_path, output_pt_path):
    """
    Extracts the adapter weights from a SafeTensor file and saves them in a PyTorch file.
    """
    full_weight = load_file(safetensor_path)
    # filter with "adapter" in the name
    adapter_weights = {k: v for k, v in full_weight.items() if "adapter" in k}
    torch.save(adapter_weights, output_pt_path)

if __name__ == "__main__":
    safetensor_path = sys.argv[1]
    output_pt_path = sys.argv[2]
    extract_weights(safetensor_path, output_pt_path)