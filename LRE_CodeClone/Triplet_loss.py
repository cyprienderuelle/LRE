import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F
import json
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_dataset(self, dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        if dataset_path.endswith('.jsonl'):
            return [json.loads(line.strip()) for line in f if line.strip()]
        else:
            try:
                f.seek(0)
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                return [json.loads(line.strip()) for line in f if line.strip()]