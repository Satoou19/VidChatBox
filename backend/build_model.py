import os
import sys

def main():
    print("==================================================")
    print("Pre-downloading SentenceTransformer model for deployment...")
    print("==================================================")
    
    model_name = "intfloat/multilingual-e5-small"
    target_dir = os.path.join("backend", "data", "models", "multilingual-e5-small")
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"Model already exists at {target_dir}. Skipping download.")
        return
        
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Downloading {model_name} from Hugging Face Hub...")
        model = SentenceTransformer(model_name)
        
        print(f"Saving model weights and config to {target_dir}...")
        model.save(target_dir)
        print("[SUCCESS] Model downloaded and saved locally in the project!")
    except Exception as e:
        print(f"[WARNING] Failed to pre-download model during build: {e}")
        print("The model will instead be downloaded dynamically at runtime if needed.")
        
    print("==================================================")

if __name__ == "__main__":
    main()
