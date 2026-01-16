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
CHUNK_SIZE = 1500 
BATCH_SIZE = 128

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
model.eval()

# --- CHARGEMENT DES TEXTES ---
print("Chargement des textes en RAM (CPU)...")
raw_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        raw_data.append(json.loads(line))

ancres_text = [d['anchor'] for d in raw_data]
total_count = len(ancres_text)

# --- TRAITEMENT PAR BLOCS AVEC BARRES DE PROGRESSION ---
if os.path.exists(output_file):
    os.remove(output_file)

# Barre de progression principale pour tout le dataset
main_pbar = tqdm(total=total_count, desc="Progression Totale", unit="anchor")

for start_idx in range(0, total_count, CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, total_count)
    
    # 1. Encodage du bloc avec sa propre barre
    chunk_texts = ancres_text[start_idx:end_idx]
    chunk_embeddings_list = []
    
    # On ajoute une barre spécifique pour l'encodage du bloc en cours
    for i in range(0, len(chunk_texts), BATCH_SIZE):
        batch = [t[:1000] for t in chunk_texts[i : i + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            vecs = torch.max(torch.log(1 + torch.relu(outputs.logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
            chunk_embeddings_list.append(vecs.cpu())
            
        # On met à jour la barre principale pendant l'encodage
        main_pbar.update(len(batch))

    chunk_embeddings = torch.cat(chunk_embeddings_list).to(device)
    
    # 2. Recherche des Hard Negatives dans le bloc
    with open(output_file, 'a', encoding='utf-8') as f_out:
        # Ajout d'une barre de progression pour la recherche intra-bloc
        search_pbar = tqdm(total=len(chunk_texts), desc=f" ↪ Recherche Bloc {start_idx//CHUNK_SIZE + 1}", leave=False)
        
        for i in range(len(chunk_texts)):
            query_vec = chunk_embeddings[i].unsqueeze(0)
            
            # Calcul de similarité
            scores = torch.mm(query_vec, chunk_embeddings.t()).squeeze(0)
            
            top_k = min(10, len(chunk_texts))
            top_indices = torch.topk(scores, k=top_k).indices.cpu().tolist()
            
            neighbors = [idx for idx in top_indices if idx != i][:5]
            neg_idx_in_chunk = random.choice(neighbors) if neighbors else random.randint(0, len(chunk_texts)-1)
            
            real_neg_idx = start_idx + neg_idx_in_chunk
            current_real_idx = start_idx + i
            
            output_obj = {
                "ancre": raw_data[current_real_idx]['anchor'],
                "pos": raw_data[current_real_idx]['positif'],
                "neg": raw_data[real_neg_idx]['anchor'],
                "type": raw_data[current_real_idx].get('type', -1)
            }
            f_out.write(json.dumps(output_obj) + '\n')
            
            # Mise à jour de la barre de recherche
            search_pbar.update(1)
            
        search_pbar.close() # On ferme la barre du bloc une fois fini
    
    # Nettoyage
    del chunk_embeddings
    del chunk_embeddings_list
    torch.cuda.empty_cache()
    gc.collect()

main_pbar.close()
print(f"\n✅ Terminé ! Ton dataset est prêt dans : {output_file}")