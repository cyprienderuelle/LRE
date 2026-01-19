import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from peft import PeftModel
# Assure-toi que SpladeTripletModel est importé correctement depuis ton fichier d'entraînement
# from Train import SpladeTripletModel 

# --- CONFIGURATION ---
path_dataset_test = "./new_piscine_dataset.jsonl"
base_path = "."
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_EVAL = 8

class SpladeTripletModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model 

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits 
        sparse_emb = torch.log1p(torch.relu(logits))
        if attention_mask is not None:
            sparse_emb = sparse_emb * attention_mask.unsqueeze(-1)
        return torch.max(sparse_emb, dim=1).values

def load_model_from_checkpoint(checkpoint_path, device):
    print(f"--- Loading Model: {checkpoint_path} ---")
    tokenizer = AutoTokenizer.from_pretrained("naver/splade_v2_max")
    base_mlm = AutoModelForMaskedLM.from_pretrained("naver/splade_v2_max") 
    
    # Chargement LoRA
    model_peft = PeftModel.from_pretrained(base_mlm, checkpoint_path)
    model = SpladeTripletModel(model_peft).to(device)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def encode_texts(texts, model, tokenizer, device):
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE_EVAL):
        batch = texts[i:i + BATCH_SIZE_EVAL]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        # Calcul SPLADE
        embeddings = model(enc['input_ids'], enc['attention_mask'])
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

def compute_ndcg_at_k(positive_ranks, k):
    """Calcul du NDCG@K pour un ensemble de rangs de positifs"""
    dcg = sum([1.0 / np.log2(r + 1) for r in positive_ranks if r <= k])
    # IDCG : cas idéal où le positif est au rang 1
    idcg = 1.0 / np.log2(1 + 1) 
    return dcg / idcg

def evaluate_on_dataset(triplets, model, tokenizer, device):
    anchors = [t['ancre'] for t in triplets]
    positives = [t['pos'] for t in triplets]
    negatives = [t['neg'] for t in triplets]

    print("Encoding anchors and candidates...")
    emb_anchors = encode_texts(anchors, model, tokenizer, device)
    emb_pos = encode_texts(positives, model, tokenizer, device)
    emb_neg = encode_texts(negatives, model, tokenizer, device)

    metrics = defaultdict(list)

    for i in range(len(emb_anchors)):
        # On simule un pool de candidats : le vrai positif + tous les négatifs du batch
        # pour rendre le test difficile (In-batch distraction)
        q_emb = emb_anchors[i].unsqueeze(0)
        
        # Scores : Produit scalaire
        pos_score = torch.mm(q_emb, emb_pos[i].unsqueeze(1)).item()
        neg_scores = torch.mm(q_emb, emb_neg.t()).squeeze(0).tolist()
        
        all_scores = [pos_score] + neg_scores
        labels = [1] + [0] * len(neg_scores) # 1 pour le positif, 0 pour les autres
        
        # Tri
        combined = sorted(zip(all_scores, labels), key=lambda x: x[0], reverse=True)
        sorted_labels = [x[1] for x in combined]
        
        # Rang du premier positif (1-indexed)
        rank = sorted_labels.index(1) + 1
        
        # Accumulation des métriques
        metrics['acc_top1'].append(1 if rank == 1 else 0)
        metrics['acc_top5'].append(1 if rank <= 5 else 0)
        metrics['mrr'].append(1.0 / rank)
        metrics['ndcg_5'].append(compute_ndcg_at_k([rank], 5))
        metrics['ndcg_10'].append(compute_ndcg_at_k([rank], 10))

    return {k: np.mean(v) for k, v in metrics.items()}

# --- EXECUTION PRINCIPALE ---
if __name__ == "__main__":
    # Charger les données de test (DeepSeek Generated)
    test_triplets = []
    with open(path_dataset_test, "r") as f:
        for line in f:
            test_triplets.append(json.loads(line))

    # Liste des checkpoints à tester
    base_path_obj = Path(base_path)
    checkpoints = sorted([d for d in base_path_obj.iterdir() if d.is_dir() and 'checkpoint' in d.name])

    final_results = []

    for cp in checkpoints:
        model, tokenizer = load_model_from_checkpoint(str(cp), device)
        res = evaluate_on_dataset(test_triplets, model, tokenizer, device)
        
        res['name'] = cp.name
        final_results.append(res)
        
        print(f"\nResults for {cp.name}:")
        print(f"Top-1: {res['acc_top1']*100:.2f}% | Top-5: {res['acc_top5']*100:.2f}%")
        print(f"MRR: {res['mrr']:.4f} | NDCG@10: {res['ndcg_10']:.4f}")

    # Sauvegarde finale pour LaTeX
    with open("final_comparison_results.json", "w") as f:
        json.dump(final_results, f, indent=4)