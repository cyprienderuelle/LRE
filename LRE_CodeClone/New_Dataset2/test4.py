import torch
import numpy as np
from transformers import (AutoModelForMaskedLM, AutoModelForCausalLM, 
                          AutoTokenizer, AutoModel, BitsAndBytesConfig)
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# --- CONFIGURATION ---
MODELS_TO_TEST = [
    {"name": "CodeBERT", "path": "microsoft/codebert-base", "type": "encoder"},
    {"name": "UniXcoder", "path": "microsoft/unixcoder-base", "type": "encoder"},
    {"name": "DeepSeek-6.7B", "path": "deepseek-ai/deepseek-coder-6.7b-base", "type": "decoder"},
    {"name": "StarCoder2-7B", "path": "bigcode/starcoder2-7b", "type": "decoder"},
]

PATH_DATASET_TEST = "./new_piscine_dataset.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CLASSE SPLADE ADAPTATIVE ---
class UniversalSpladeModel(torch.nn.Module):
    def __init__(self, base_model, model_type="encoder"):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extraire les logits selon le type de modèle
        if self.model_type == "encoder":
            # Pour MaskedLM (CodeBERT)
            logits = outputs.logits
        else:
            # Pour CausalLM (DeepSeek/StarCoder)
            logits = outputs.logits
            
        # SPLADE pooling: log(1 + ReLU(w))
        values, _ = torch.max(torch.log(1 + torch.relu(logits)), dim=1)
        return values

# --- FONCTIONS DE CHARGEMENT ---
def load_any_model(model_info, device):
    print(f"\n🚀 Chargement de {model_info['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_info['path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_info['type'] == "encoder":
        # On utilise MaskedLM pour avoir la tête de vocabulaire
        base = AutoModelForMaskedLM.from_pretrained(model_info['path']).to(device)
    else:
        # Pour les gros modèles 6B+, on utilise la quantification 4-bit si possible
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_info['path'],
            quantization_config=bnb_config if device == "cuda" else None,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

    model = UniversalSpladeModel(base, model_info['type'])
    model.eval()
    return model, tokenizer

# --- LOGIQUE D'ÉVALUATION (OPTIMISÉE) ---
@torch.no_grad()
def encode_batch(texts, model, tokenizer, device, batch_size=4):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        emb = model(enc['input_ids'], enc['attention_mask'])
        all_embeddings.append(emb.cpu())
        
    return torch.cat(all_embeddings, dim=0)

def compute_metrics(triplets_by_type, model, tokenizer, device):
    all_mrr = []
    all_acc = []
    
    for type_name, triplets in triplets_by_type.items():
        # Préparation des textes
        anchors = [t["ancre"] for t in triplets]
        positives = list(set([t["pos"] for t in triplets]))
        negatives = list(set([t["neg"] for t in triplets]))
        candidates = positives + negatives
        labels = [1]*len(positives) + [0]*len(negatives)

        # Encodage
        emb_anchors = encode_batch(anchors, model, tokenizer, device)
        emb_cands = encode_batch(candidates, model, tokenizer, device)

        # Calcul des scores (Produit scalaire)
        for i in range(len(emb_anchors)):
            scores = torch.matmul(emb_anchors[i], emb_cands.T)
            sorted_idx = torch.argsort(scores, descending=True).numpy()
            sorted_labels = [labels[idx] for idx in sorted_idx]

            # Acc @1
            all_acc.append(1.0 if sorted_labels[0] == 1 else 0.0)
            # MRR
            try:
                rank = sorted_labels.index(1) + 1
                all_mrr.append(1.0 / rank)
            except: all_mrr.append(0.0)

    return {"accuracy": np.mean(all_acc), "mrr": np.mean(all_mrr)}

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    # 1. Charger le dataset
    triplets_by_type = defaultdict(list)
    with open(PATH_DATASET_TEST, "r") as f:
        for line in f:
            data = json.loads(line)
            triplets_by_type[data["function_type"]].append(data)

    results = {}

    # 2. Boucle sur les modèles
    for m_info in MODELS_TO_TEST:
        try:
            model, tokenizer = load_any_model(m_info, DEVICE)
            metrics = compute_metrics(triplets_by_type, model, tokenizer, DEVICE)
            results[m_info['name']] = metrics
            
            print(f"✅ {m_info['name']} -> Acc: {metrics['accuracy']:.4f}, MRR: {metrics['mrr']:.4f}")
            
            # Nettoyage mémoire critique pour DeepSeek/StarCoder
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Erreur avec {m_info['name']}: {e}")

    # 3. Visualisation comparative
    names = list(results.keys())
    accs = [r['accuracy'] for r in results.values()]
    mrrs = [r['mrr'] for r in results.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, accs, width, label='Accuracy', color='#2E86AB')
    ax.bar(x + width/2, mrrs, width, label='MRR', color='#A23B72')

    ax.set_ylabel('Scores')
    ax.set_title('Comparaison des Modèles sur le dataset Piscine')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    
    plt.savefig("comparaison_modeles.png")
    plt.show()