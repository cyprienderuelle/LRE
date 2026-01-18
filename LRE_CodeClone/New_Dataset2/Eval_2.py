import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F

# Import de ta classe Splade
from Train import SpladeTripletModel

# --- CONFIG ---
MODEL_ID = "naver/splade_v2_max"
# Liste de tes dossiers contenant les checkpoints Full Fine-Tuning (pas des adapters LoRA)
checkpoint_list = ["./checkpoint_full_epoch_v3_1", "./checkpoint_full_epoch_v3_2"] 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
SAMPLE_SIZE = 2000

def run_evaluation(model, tokenizer, codes, labels, desc="Evaluation"):
    model.eval()
    
    def get_embeddings(texts):
        embeddings = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Encodage {desc}"):
            batch = [str(t)[:1000] for t in texts[i : i + BATCH_SIZE]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            
            with torch.no_grad():
                # Utilisation de ton architecture Splade
                vecs = model(inputs['input_ids'], inputs['attention_mask'])
                embeddings.append(vecs.cpu())
        return torch.cat(embeddings)

    all_vecs = get_embeddings(codes)
    
    print(f"📊 Calcul MAP pour {desc}...")
    # Calcul de la matrice de similarité par produit scalaire (standard pour SPLADE)
    sim_matrix = torch.mm(all_vecs, all_vecs.t())
    # On ignore l'auto-similarité
    sim_matrix.fill_diagonal_(-1e9)

    aps = []
    for i in range(len(labels)):
        target_label = labels[i]
        scores = sim_matrix[i]
        sorted_indices = torch.argsort(scores, descending=True).numpy()
        is_clone = (labels[sorted_indices] == target_label)
        
        if np.sum(is_clone) > 0:
            hits = np.cumsum(is_clone)
            prec = hits / (np.arange(len(is_clone)) + 1)
            aps.append(np.sum(prec * is_clone) / np.sum(is_clone))
    
    return np.mean(aps)

# 1. Préparation des données (POJ-104 pour le langage C)
print("📦 Chargement de POJ-104...")
dataset = load_dataset("google/code_x_glue_cc_clone_detection_poj104", split="test")
all_codes = dataset['code'][:SAMPLE_SIZE]
all_labels = np.array(dataset['label'][:SAMPLE_SIZE])

# 2. TEST BASELINE (Modèle original Naver)
print("\n--- TEST 1: BASELINE (Modèle original) ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_mlm = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
model_baseline = SpladeTripletModel(base_mlm).to(DEVICE)

map_baseline = run_evaluation(model_baseline, tokenizer, all_codes, all_labels, desc="Baseline")
print(f"❌ MAP Baseline: {map_baseline:.4f}")

# Nettoyage mémoire vive/vidéo
del model_baseline, base_mlm
torch.cuda.empty_cache()

results = {"Baseline": map_baseline}

# 3. TEST DES CHECKPOINTS FULL FINE-TUNING
for CHECKPOINT_PATH in checkpoint_list:
    print(f"\n--- TEST: {CHECKPOINT_PATH} (Full Fine-Tuned) ---")
    
    # On charge le modèle complet directement depuis le dossier
    # Plus besoin de PeftModel, car les poids sont déjà fusionnés/complets
    ft_mlm = AutoModelForMaskedLM.from_pretrained(CHECKPOINT_PATH)
    model_ft = SpladeTripletModel(ft_mlm).to(DEVICE)

    map_ft = run_evaluation(model_ft, tokenizer, all_codes, all_labels, desc="FT Model")
    results[CHECKPOINT_PATH] = map_ft
    print(f"✅ MAP FT Model: {map_ft:.4f}")

    del model_ft, ft_mlm
    torch.cuda.empty_cache()

# --- RÉSUMÉ ---
print("\n" + "="*40)
print(f"{'Modèle':<25} | {'MAP':<10}")
print("-" * 40)
for name, m_value in results.items():
    print(f"{name:<25} | {m_value:.4f}")
print("="*40)