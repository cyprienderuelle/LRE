import json
import torch
import random
import gc
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# --- CONFIGURATION ---
base_model_id = "naver/splade_v2_max"
lora_checkpoint_path = "./checkpoint_epoch_map70" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_file = 'Resultat.jsonl'
output_file = 'Dataset_InfoNCE_HardNeg_V3.jsonl'
CHUNK_SIZE = 50000
BATCH_SIZE = 128

# 1. Chargement
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForMaskedLM.from_pretrained(base_model_id)
print(f"Chargement du modèle fine-tuné depuis {lora_checkpoint_path}...")
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path).to(device)
model.eval()

# --- CHARGEMENT DES TEXTES ---
print("Chargement des textes en RAM (CPU)...")
raw_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        raw_data.append(json.loads(line))

total_count = len(raw_data)

# --- TRAITEMENT PAR BLOCS ---
if os.path.exists(output_file):
    os.remove(output_file)

# La barre principale suivra chaque ancre générée
main_pbar = tqdm(total=total_count, desc="Total", unit="anchor")

def encode_list(text_list, desc):
    embeddings = []
    # Barre secondaire pour l'encodage GPU
    for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc=f"   ↳ {desc}", leave=False):
        batch = [str(t)[:1000] for t in text_list[i : i + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            vecs = torch.max(torch.log1p(torch.relu(outputs.logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
            embeddings.append(vecs.cpu())
    return torch.cat(embeddings)

for start_idx in range(0, total_count, CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, total_count)
    chunk_data = raw_data[start_idx:end_idx]
    
    # Encodage
    chunk_anchors = [d['anchor'] for d in chunk_data]
    chunk_positives = [d['positif'] for d in chunk_data]
    
    all_anchor_vecs = encode_list(chunk_anchors, "Encodage Ancres")
    all_pos_vecs = encode_list(chunk_positives, "Encodage Positifs")
    
    # Recherche et Écriture
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i in range(len(chunk_data)):
            current_real_idx = start_idx + i
            target_type = raw_data[current_real_idx].get('type')
            
            # Calcul de similarité (Positif vs Ancres du bloc)
            current_pos_vec = all_pos_vecs[i].unsqueeze(0).to(device)
            scores = torch.mm(current_pos_vec, all_anchor_vecs.to(device).t()).squeeze(0).cpu()
            
            # On prend un top_k large pour filtrer les types identiques
            top_k = min(50, len(chunk_data))
            top_indices = torch.topk(scores, k=top_k).indices.tolist()
            
            # Filtrage : On veut un hard negative qui n'est PAS du même type
            neighbors = [idx for idx in top_indices if idx != i and raw_data[start_idx + idx].get('type') != target_type][:5]
            
            if neighbors:
                neg_idx_in_chunk = random.choice(neighbors)
            else:
                neg_idx_in_chunk = random.randint(0, len(chunk_data)-1)
            
            real_neg_idx = start_idx + neg_idx_in_chunk
            
            output_obj = {
                "ancre": raw_data[current_real_idx]['anchor'],
                "pos": raw_data[current_real_idx]['positif'],
                "neg": raw_data[real_neg_idx]['anchor'],
                "type": target_type
            }
            f_out.write(json.dumps(output_obj) + '\n')
            
            # MISE À JOUR DE LA BARRE PRINCIPALE
            main_pbar.update(1)
    
    # Nettoyage mémoire après chaque bloc
    del all_anchor_vecs, all_pos_vecs, chunk_data
    torch.cuda.empty_cache()
    gc.collect()

main_pbar.close()
print(f"\n✅ Terminé ! Nouveau dataset filtré généré.")