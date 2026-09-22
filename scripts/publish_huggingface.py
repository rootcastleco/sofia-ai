"""Automated release publisher for Sofia Engine to Hugging Face.

Selectively uploads approved artifacts, manifests, documentation, and examples
to the Hugging Face repository `rootcastleengineering/sofia`.

Requires:
    export HF_TOKEN="hf_..."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub is required. Install with: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

HF_REPO_ID = "rootcastleengineering/sofia"
STAGING_DIR = Path(__file__).resolve().parent.parent / "dist" / "hf-repo"


def publish_to_huggingface() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable is not set.", file=sys.stderr)
        print("Set HF_TOKEN with write access to rootcastleengineering/sofia.", file=sys.stderr)
        sys.exit(1)

    if not STAGING_DIR.exists():
        print(f"Error: Staging directory not found: {STAGING_DIR}", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    print(f"Publishing approved artifacts from {STAGING_DIR} to https://huggingface.co/{HF_REPO_ID}...")

    # Upload folder selectively (excluding .git)
    api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="release: publish Sofia Engine 3.0.0a1 artifacts and manifests",
        ignore_patterns=[".git", ".git/**"],
    )

    print(f"Successfully published release artifacts to https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    publish_to_huggingface()
