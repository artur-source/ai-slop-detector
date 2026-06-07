# Application configuration values shared by training and the Flask app.

from dataclasses import dataclass

import torch


@dataclass
class AppConfig:
    """Configuration defaults for the AI slop detector project."""

    model_name: str = "distilbert-base-uncased"
    saved_model_path: str = "model/saved"
    max_length: int = 512
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    train_ratio: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    flask_host: str = "0.0.0.0"
    flask_port: int = 5000
    min_text_length: int = 100
    max_text_length: int = 5000

