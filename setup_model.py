import os
import shutil
from huggingface_hub import snapshot_download

def setup_model():
    """
    Download and setup the SpeechBrain model without symlinks
    """
    model_dir = "pretrained_models/spkrec_ecapa"
    repo_id = "speechbrain/spkrec-ecapa-voxceleb"
    
    print(f"Downloading {repo_id} to {model_dir}...")
    
    # Create the directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Download the model files directly
    snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False  # This is the key - don't use symlinks
    )
    
    print("Model downloaded successfully!")
    print("Files in model directory:")
    for f in os.listdir(model_dir):
        print(f"  {f}")

if __name__ == "__main__":
    try:
        setup_model()
    except Exception as e:
        print(f"Error setting up model: {e}")
        import traceback
        traceback.print_exc()