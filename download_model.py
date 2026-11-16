import os
from huggingface_hub import hf_hub_download
import safetensors.torch

def download_and_verify(save_to_local_dir=True):
    """
    Mengunduh bobot model ArcFace R50 FP32 dari Hugging Face repo 'py-feat/arcface_r50'.
    """
    repo_id = "py-feat/arcface_r50"
    filename = "arcface_r50.safetensors"
    
    print(f"Mengunduh {filename} dari {repo_id}...")
    
    if save_to_local_dir:
        local_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(local_dir, exist_ok=True)
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    else:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    print(f"File berhasil diunduh ke: {model_path}")
    
    # Verifikasi isi bobot model
    state_dict = safetensors.torch.load_file(model_path)
    sample_key = next((k for k in state_dict.keys() if "weight" in k), None)
    
    print("\n--- Verifikasi Model ---")
    print(f"Total parameter/tensor : {len(state_dict)}")
    if sample_key:
        print(f"Sample tensor         : {sample_key}")
        print(f"Tipe data (dtype)     : {state_dict[sample_key].dtype}")
        print(f"Ukuran tensor         : {state_dict[sample_key].shape}")
    print("------------------------\n")
    return model_path

if __name__ == "__main__":
    download_and_verify(save_to_local_dir=True)
