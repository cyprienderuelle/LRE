import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import torch.nn.functional as F

from Train import SpladeTripletModel

# --- CONFIG ---
MODEL_ID = "naver/splade_v2_max"
lora_list = ["./checkpoint_epoch_v2_1", "./checkpoint_epoch_v2_2", "./checkpoint_epoch_v2_3"] # Ton dossier contenant adapter_model.bin
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
                # Utilise ta classe SpladeTripletModel
                vecs = model(inputs['input_ids'], inputs['attention_mask'])
                embeddings.append(vecs.cpu())
        return torch.cat(embeddings)

    all_vecs = get_embeddings(codes)
    
    print(f"📊 Calcul MAP pour {desc}...")
    sim_matrix = torch.mm(all_vecs, all_vecs.t())
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

# 1. Préparation des données
print("📦 Chargement de POJ-104...")
dataset = load_dataset("google/code_x_glue_cc_clone_detection_poj104", split="test")
all_codes = dataset['code'][:SAMPLE_SIZE]
all_labels = np.array(dataset['label'][:SAMPLE_SIZE])

# 2. TEST BASELINE (Modèle original)
print("\n--- TEST 1: BASELINE (Modèle original) ---")
base_mlm = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model_baseline = SpladeTripletModel(base_mlm).to(DEVICE)

map_baseline = run_evaluation(model_baseline, tokenizer, all_codes, all_labels, desc="Baseline")
print(f"❌ MAP Baseline: {map_baseline:.4f}")

# Nettoyage mémoire
del model_baseline, base_mlm
torch.cuda.empty_cache()

for LORA_PATH in lora_list:
    # 3. TEST MODÈLE FINE-TUNÉ
    print("\n--- TEST 2: LORA MODEL (Fine-tuné) ---")
    base_mlm_for_lora = AutoModelForMaskedLM.from_pretrained(MODEL_ID)

    print(f"Chargement de l'adapteur LoRA depuis {LORA_PATH}...")
    model_peft = PeftModel.from_pretrained(base_mlm_for_lora, LORA_PATH)
    # On l'encapsule dans Splade
    model_lora = SpladeTripletModel(model_peft).to(DEVICE)

    map_lora = run_evaluation(model_lora, tokenizer, all_codes, all_labels, desc="LoRA Model")
    print(f"✅ MAP LoRA Model: {map_lora:.4f}")

# --- RÉSUMÉ ---
print("\n" + "="*30)
print(f"RÉSULTATS FINAUX (MAP)")
print(f"Baseline:  {map_baseline:.4f}")
print(f"LoRA:      {map_lora:.4f}")
print(f"Amélioration: {((map_lora/map_baseline)-1)*100:.1f}%" if map_baseline > 0 else "N/A")
print("="*30)