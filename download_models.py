import os
import sys

# Disable tqdm output globally to avoid the Windows RLock recursion bug
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def cache_models():
    print("Initiating clean model caching process...")
    from huggingface_hub import snapshot_download
    
    print("1/2 Downloading paraphrase-multilingual-MiniLM-L12-v2 (approx 470 MB)")
    snapshot_download(repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", local_files_only=False)
    print("MiniLM downloaded successfully!")
    
    print("2/2 Downloading google/mt5-small (approx 1.2 GB)")
    snapshot_download(repo_id="google/mt5-small", local_files_only=False)
    print("MT5-small downloaded successfully!")

    print("All models successfully cached! It is now safe to start Streamlit.")

if __name__ == "__main__":
    cache_models()
