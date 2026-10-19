"""Hugging Face Spaces entrypoint; the Space installs requirements before copying this repo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from finance_sentiment_slm_finetune.app import ALL_INTERFACES, main  # noqa: E402

if __name__ == "__main__":
    main(server_name=ALL_INTERFACES)
