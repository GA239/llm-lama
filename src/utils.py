import os

from huggingface_hub import snapshot_download


def download_model(repo_id, local_dir, **kwargs):
    snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.join(local_dir, repo_id),
        **kwargs,
    )
