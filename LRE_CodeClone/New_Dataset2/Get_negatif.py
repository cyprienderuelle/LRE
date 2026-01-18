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
lora_checkpoint_path = "./checkpoint_epoch_map70" 
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
    
    chunk_data = raw_data[start_idx:end_idx]
    
    # 1. On a besoin des deux listes pour comparer Positif vs les autres Ancres
    chunk_anchors = [d['anchor'] for d in chunk_data]
    chunk_positives = [d['positif'] for d in chunk_data]
    
    def encode_list(text_list, desc):
        embeddings = []
        for i in range(0, len(text_list), BATCH_SIZE):
            batch = [str(t)[:1000] for t in text_list[i : i + BATCH_SIZE]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                vecs = torch.max(torch.log1p(torch.relu(outputs.logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
                embeddings.append(vecs.cpu())
        return torch.cat(embeddings)

    print(f"\n🚀 Encodage du Bloc {start_idx//CHUNK_SIZE + 1}...")
    # On encode les ancres (qui serviront de réservoir de négatifs)
    all_anchor_vecs = encode_list(chunk_anchors, "Anchors").to(device)
    # On encode les positifs (qui serviront de requêtes pour trouver des négatifs proches)
    all_pos_vecs = encode_list(chunk_positives, "Positives").to(device)
    
    # 2. Recherche des Hard Negatives (Proches du POSITIF)
    with open(output_file, 'a', encoding='utf-8') as f_out:
        search_pbar = tqdm(total=len(chunk_data), desc=" ↪ Recherche par proximité Positif", leave=False)
        
        for i in range(len(chunk_data)):
            # On prend le vecteur du POSITIF actuel
            current_pos_vec = all_pos_vecs[i].unsqueeze(0)
            
            # On calcule sa similarité avec TOUTES les ANCRES du bloc
            # (Le but: trouver une ancre d'un autre problème qui ressemble à mon code positif actuel)
            scores = torch.mm(current_pos_vec, all_anchor_vecs.t()).squeeze(0)
            
            # On prend les 15 plus proches
            top_k = min(15, len(chunk_data))
            top_indices = torch.topk(scores, k=top_k).indices.cpu().tolist()
            
            # On exclut l'indice i (car l'ancre i est liée au positif i)
            # On prend parmi les indices 0 à 5 du top (les plus "fourbes")
            neighbors = [idx for idx in top_indices if idx != i][0:5]
            
            if neighbors:
                neg_idx_in_chunk = random.choice(neighbors)
            else:
                neg_idx_in_chunk = random.randint(0, len(chunk_data)-1)
            
            # Enregistrement
            real_neg_idx = start_idx + neg_idx_in_chunk
            current_real_idx = start_idx + i
            
            output_obj = {
                "ancre": raw_data[current_real_idx]['anchor'],
                "pos": raw_data[current_real_idx]['positif'],
                "neg": raw_data[real_neg_idx]['anchor'], # C'est l'ancre la plus proche de notre positif
                "type": raw_data[current_real_idx].get('type', -1)
            }
            f_out.write(json.dumps(output_obj) + '\n')
            search_pbar.update(1)
            
        search_pbar.close()
    
    del chunk_anchors
    del chunk_positives
    torch.cuda.empty_cache()
    gc.collect()

main_pbar.close()
print(f"\n✅ Terminé ! Nouveau dataset généré avec le modèle fine-tuné.")