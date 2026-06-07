# Data

This project uses the HC3 dataset for training and evaluation.

You can load the dataset through HuggingFace `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("Hello-SimpleAI/HC3")
```

Direct dataset link: https://huggingface.co/datasets/Hello-SimpleAI/HC3

The dataset download happens automatically in `model/train_classifier.py`, so you do not need to manually place files in this directory before training.

Note: the dataset is approximately 50MB, depending on cache format and selected split.

