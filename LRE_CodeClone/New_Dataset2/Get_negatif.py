import json
import torch
import random
import gc
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel # <--- Import nécessaire pour charger ton LoRA
from tqdm import tqdm

# --- CONFIGURATION ---
base_model_id = "naver/splade_v2_max"
lora_checkpoint_path = "./checkpoint_epoch_1" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_file = 'Resultat.jsonl'
output_file = 'Dataset_InfoNCE_HardNeg_V3.jsonl'
CHUNK_SIZE = 50000
BATCH_SIZE = 128

# 1. Chargement du Tokenizer et du modèle de base
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForMaskedLM.from_pretrained(base_model_id)

# 2. Chargement des adaptateurs LoRA (Ton modèle à 81%)
print(f"Chargement du modèle fine-tuné depuis {lora_checkpoint_path}...")
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path).to(device)
model.eval()

# --- CHARGEMENT DES TEXTES ---
print("Chargement des textes en RAM (CPU)...")
raw_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        raw_data.append(json.loads(line))

ancres_text = [d['anchor'] for d in raw_data]
total_count = len(ancres_text)

# --- TRAITEMENT PAR BLOCS ---
if os.path.exists(output_file):
    os.remove(output_file)

main_pbar = tqdm(total=total_count, desc="Progression Totale", unit="anchor")

for start_idx in range(0, total_count, CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, total_count)
    
    chunk_texts = ancres_text[start_idx:end_idx]
    chunk_embeddings_list = []
    
    for i in range(0, len(chunk_texts), BATCH_SIZE):
        batch = [t[:1000] for t in chunk_texts[i : i + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            # Attention : PeftModel renvoie la structure de base (MaskedLM)
            outputs = model(**inputs)
            # Calcul SPLADE standard : log(1 + ReLU(logits))
            # On n'utilise PLUS la normalisation L2 pour être cohérent avec ton entraînement
            vecs = torch.max(torch.log1p(torch.relu(outputs.logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
            chunk_embeddings_list.append(vecs.cpu())
            
        main_pbar.update(len(batch))

    chunk_embeddings = torch.cat(chunk_embeddings_list).to(device)
    
    # Recherche des Hard Negatives
    with open(output_file, 'a', encoding='utf-8') as f_out:
        search_pbar = tqdm(total=len(chunk_texts), desc=f" ↪ Recherche Bloc {start_idx//CHUNK_SIZE + 1}", leave=False)
        
        for i in range(len(chunk_texts)):
            query_vec = chunk_embeddings[i].unsqueeze(0)
            
            # Calcul de similarité via Produit Scalaire (Dot Product)
            # C'est ici que ton modèle à 81% va briller pour trouver des négatifs difficiles
            scores = torch.mm(query_vec, chunk_embeddings.t()).squeeze(0)
            
            top_k = min(15, len(chunk_texts)) # On cherche un peu plus loin (15 au lieu de 10)
            top_indices = torch.topk(scores, k=top_k).indices.cpu().tolist()
            
            # On évite l'ancre elle-même (i) et on prend parmi les plus proches (indices 1 à 6)
            neighbors = [idx for idx in top_indices if idx != i][0:5]
            
            if neighbors:
                neg_idx_in_chunk = random.choice(neighbors)
            else:
                neg_idx_in_chunk = random.randint(0, len(chunk_texts)-1)
            
            real_neg_idx = start_idx + neg_idx_in_chunk
            current_real_idx = start_idx + i
            
            output_obj = {
                "ancre": raw_data[current_real_idx]['anchor'],
                "pos": raw_data[current_real_idx]['positif'],
                "neg": raw_data[real_neg_idx]['anchor'],
                "type": raw_data[current_real_idx].get('type', -1)
            }
            f_out.write(json.dumps(output_obj) + '\n')
            search_pbar.update(1)
            
        search_pbar.close()
    
    del chunk_embeddings
    del chunk_embeddings_list
    torch.cuda.empty_cache()
    gc.collect()

main_pbar.close()
print(f"\n✅ Terminé ! Nouveau dataset généré avec le modèle fine-tuné.")