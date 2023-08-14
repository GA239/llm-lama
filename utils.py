from huggingface_hub import snapshot_download
import os


def download_model(repo_id, local_dir):
    snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.join(local_dir, repo_id)
    )