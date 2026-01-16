import json
import torch
import random
import gc
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
model_id = "naver/splade_v2_max"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_file = 'Resultat.jsonl'
output_file = 'Dataset_InfoNCE_HardNeg.jsonl'
CHUNK_SIZE = 50000  # On traite 50k fonctions à la fois pour épargner la RAM

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
model.eval()

def get_splade_embeddings_batch(texts, batch_size=128):
    """ Calcule les vecteurs SPLADE pour un bloc donné """
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t[:1000] for t in texts[i : i + batch_size]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            # Logique SPLADE
            vecs = torch.max(torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
            
            all_embeddings.append(vecs.cpu()) # On stocke sur CPU pendant l'encodage
            
    return torch.cat(all_embeddings)

# --- CHARGEMENT DES TEXTES ---
print("Chargement des textes en RAM (CPU)...")
raw_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        raw_data.append(json.loads(line))

ancres_text = [d['anchor'] for d in raw_data]
total_count = len(ancres_text)
print(f"Total à traiter : {total_count} fonctions.")

# --- TRAITEMENT PAR BLOCS ---
if os.path.exists(output_file):
    os.remove(output_file) # On repart sur un fichier propre

for start_idx in range(0, 50000, CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, total_count)
    print(f"\n--- Traitement du bloc {start_idx} à {end_idx} ---")
    
    # 1. Encodage du bloc actuel
    chunk_texts = ancres_text[start_idx:end_idx]
    chunk_embeddings = get_splade_embeddings_batch(chunk_texts).to(device)
    
    # 2. On a besoin de comparer ce bloc à TOUT le dataset.
    # Pour éviter de tout encoder, on compare le bloc contre lui-même (ou on ré-encode 
    # une partie aléatoire si on veut des négatifs plus larges).
    # OPTION : Ici on cherche les négatifs AU SEIN du bloc actuel pour garantir la RAM.
    
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i in tqdm(range(len(chunk_texts)), desc="Recherche Hard Negatives (Bloc)"):
            query_vec = chunk_embeddings[i].unsqueeze(0)
            
            # Similarité à l'intérieur du bloc GPU
            scores = torch.mm(query_vec, chunk_embeddings.t()).squeeze(0)
            
            # Top 10
            top_k = min(10, len(chunk_texts))
            top_indices = torch.topk(scores, k=top_k).indices.cpu().tolist()
            
            # Filtrer l'ancre elle-même
            neighbors = [idx for idx in top_indices if idx != i][:5]
            
            if not neighbors:
                neg_idx_in_chunk = random.randint(0, len(chunk_texts)-1)
            else:
                neg_idx_in_chunk = random.choice(neighbors)
            
            # Index réel dans raw_data
            real_neg_idx = start_idx + neg_idx_in_chunk
            
            # Construction de l'objet
            current_real_idx = start_idx + i
            output_obj = {
                "ancre": raw_data[current_real_idx]['anchor'],
                "pos": raw_data[current_real_idx]['positif'],
                "neg": raw_data[real_neg_idx]['anchor'],
                "type": raw_data[current_real_idx].get('type', -1)
            }
            f_out.write(json.dumps(output_obj) + '\n')
    
    # 3. NETTOYAGE CRUCIAL DE LA MÉMOIRE
    del chunk_embeddings
    torch.cuda.empty_cache()
    gc.collect()

print("\n--- Terminé ! Fichier Dataset_InfoNCE_HardNeg.jsonl généré. ---")