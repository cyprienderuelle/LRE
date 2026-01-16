import json
import torch
import random
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
model_id = "naver/splade_v2_max"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
model.eval()

def get_splade_embeddings(texts, batch_size=32):
    """ Calcule les vecteurs SPLADE pour une liste de textes """
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding SPLADE"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Logique SPLADE : max sur la dimension sequence, activation ReLU et log
            logits = outputs.logits
            vecs = torch.max(torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
        
        all_embeddings.append(vecs.cpu())
    
    return torch.cat(all_embeddings)

# --- CHARGEMENT DES DONNÉES ---
ancres = []
with open('Resultat.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        ancres.append(json.loads(line)['anchor'])
        if len(ancres) >= 10000: break # Test sur 10k pour commencer

# --- CALCUL ET RECHERCHE ---

# 1. Encodage de toutes les ancres
embeddings = get_splade_embeddings(ancres).to(device)

def find_random_top5_neighbor(query_idx, embeddings, top_k=6): # 6 car l'ancre elle-même sera top 1
    # Calcul de la similarité cosinus (vecteurs normalisés)
    query_vec = embeddings[query_idx].unsqueeze(0)
    
    # Normalisation pour cos_sim
    query_vec = torch.nn.functional.normalize(query_vec, p=2, dim=1)
    dataset_vecs = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Produit matriciel pour obtenir les scores
    scores = torch.mm(query_vec, dataset_vecs.t()).squeeze(0)
    
    # Récupération des indices des top_k
    top_indices = torch.topk(scores, k=top_k).indices.cpu().tolist()
    
    # On retire l'indice de la requête elle-même (souvent le premier)
    neighbors = [idx for idx in top_indices if idx != query_idx]
    
    # On prend les 5 premiers restants et on en choisit un au hasard
    selected_neighbor_idx = random.choice(neighbors[:5])
    return selected_neighbor_idx, scores[selected_neighbor_idx].item()

# --- EXEMPLE D'UTILISATION ---
idx_test = random.randint(0, len(ancres)-1)
neighbor_idx, score = find_random_top5_neighbor(idx_test, embeddings)

print(f"\n--- ANCRE ORIGINALE (Index {idx_test}) ---")
print(ancres[idx_test][:200] + "...")

print(f"\n--- VOISIN SÉLECTIONNÉ (Index {neighbor_idx}, Score SPLADE: {score:.4f}) ---")
print(ancres[neighbor_idx][:200] + "...")