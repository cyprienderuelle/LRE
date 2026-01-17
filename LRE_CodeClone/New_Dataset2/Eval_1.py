import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- CONFIGURATION ---
MODEL_ID = "naver/splade_v2_max"
LORA_PATH = "./checkpoint_epoch_1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# 1. Chargement du modèle SPLADE fine-tuné
print("🚀 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
model.eval()

# 2. Chargement du dataset POJ-104 (Test set uniquement pour aller vite)
print("📦 Chargement de POJ-104 (Test set)...")
dataset = load_dataset("piraul/poj-104", split="test")

def get_embeddings(texts):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # Calcul SPLADE : max(log(1 + relu(logits)))
            vecs = torch.max(torch.log1p(torch.relu(outputs.logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
            embeddings.append(vecs.cpu())
    return torch.cat(embeddings)

# 3. Encodage de tous les codes du benchmark
print(f"🧬 Encodage de {len(dataset)} programmes C...")
all_codes = dataset['code']
all_labels = np.array(dataset['label'])
all_vecs = get_embeddings(all_codes)

# 4. Calcul du MAP (Mean Average Precision)
print("📊 Calcul des performances de détection de clones...")

def calculate_map(embeddings, labels):
    # Calcul de la matrice de similarité (Dot Product)
    # [N, N]
    sim_matrix = torch.mm(embeddings, embeddings.t())
    # On met la diagonale à -inf pour ne pas se trouver soi-même
    sim_matrix.fill_diagonal_(-float('inf'))
    
    avg_precisions = []
    
    # On teste pour chaque code
    for i in tqdm(range(len(labels)), desc="Evaluation"):
        label = labels[i]
        scores = sim_matrix[i]
        
        # On trie les indices par score décroissant
        sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
        
        # On regarde où se trouvent les "vrais" clones (même label)
        is_clone = (labels[sorted_indices] == label)
        
        # Calcul de la précision moyenne (Average Precision)
        num_clones = np.sum(is_clone)
        if num_clones == 0: continue
        
        hits = np.cumsum(is_clone)
        precisions = hits / (np.arange(len(is_clone)) + 1)
        ap = np.sum(precisions * is_clone) / num_clones
        avg_precisions.append(ap)
        
    return np.mean(avg_precisions)

map_score = calculate_map(all_vecs, all_labels)

print(f"\n" + "="*30)
print(f"✅ RÉSULTATS SUR POJ-104")
print(f"MAP Score : {map_score:.4f}")
print(f"="*30)