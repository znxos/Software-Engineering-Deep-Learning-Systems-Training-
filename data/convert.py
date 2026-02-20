import torch
import json
import gzip
import sys
import os

def convert_pytorch_to_json(pytorch_path, output_path):
    print(f"Loading PyTorch model from {pytorch_path}...")
    try:
        # Load state dict
        state_dict = torch.load(pytorch_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Converting tensors to list format (this may take a while)...")
    
    # Burn expects a nested dictionary structure or flat keys depending on the recorder.
    # JsonGzFileRecorder usually expects a structure matching the module hierarchy.
    # This is a simple dump; key mapping might still be needed depending on exact architecture match.
    
    json_data = {}
    
    for key, tensor in state_dict.items():
        # Convert tensor to list
        # Note: This creates huge files. For production, use binary formats.
        # We flatten the tensor data for simplicity in JSON
        data = tensor.detach().cpu().numpy().tolist()
        json_data[key] = data

    print(f"Saving to {output_path}...")
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        json.dump(json_data, f)
    
    print("Done! You can now try loading this file with --pretrained-path")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <pytorch_model.bin> [output.json.gz]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "model.json.gz"
        convert_pytorch_to_json(input_file, output_file)