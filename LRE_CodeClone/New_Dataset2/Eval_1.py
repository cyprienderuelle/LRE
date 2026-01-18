import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from New_Dataset2.Train import SpladeTripletModel

# --- CONFIG ---
MODEL_ID = "naver/splade_v2_max"
LORA_PATH = "./checkpoint_epoch_1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

def load_base_model_only(device):
    # On définit le nom du modèle original sur le Hub HuggingFace
    model_id = "naver/splade_v2_max"
    
    print(f"Loading ORIGINAL base model (no fine-tuning): {model_id}")
    
    # 1. Charger le tokenizer original
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. Charger le modèle MLM original (poids de base uniquement)
    base_mlm = AutoModelForMaskedLM.from_pretrained(model_id)
    
    # 3. L'emballer dans ta structure SPLADE (pour avoir la méthode forward correcte)
    model = SpladeTripletModel(base_mlm).to(device)
    
    model.eval()
    return model, tokenizer

print("🚀 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
# model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
model, tokenizer = load_base_model_only(DEVICE)

# LE NOUVEAU NOM 2026 (Format Parquet garanti)
print("📦 Chargement de POJ-104 (Format Google/Parquet)...")
dataset = load_dataset("google/code_x_glue_cc_clone_detection_poj104", split="test")

# Extraction des colonnes (Vérifiées pour ce dataset)
all_codes = dataset['code']
all_labels = np.array(dataset['label'])

# On limite à 2000 pour ne pas saturer la RAM au premier test
sample_size = min(2000, len(all_codes))
all_codes = all_codes[:sample_size]
all_labels = all_labels[:sample_size]

def get_embeddings(texts):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encodage"):
        batch = [str(t)[:1000] for t in texts[i : i + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        
        with torch.no_grad():
            # Utilise directement l'appel au modèle SPLADE
            # Ta classe SpladeTripletModel fait déjà le calcul du vecteur sparse
            vecs = model(inputs['input_ids'], inputs['attention_mask'])
            embeddings.append(vecs.cpu())
            
    return torch.cat(embeddings)

print(f"🧬 Encodage de {sample_size} programmes C...")
all_vecs = get_embeddings(all_codes)

# --- CALCUL MAP ---
print("📊 Calcul du score...")
sim_matrix = torch.mm(all_vecs, all_vecs.t())
sim_matrix.fill_diagonal_(-1e9)

aps = []
for i in range(len(all_labels)):
    target_label = all_labels[i]
    scores = sim_matrix[i]
    sorted_indices = torch.argsort(scores, descending=True).numpy()
    
    is_clone = (all_labels[sorted_indices] == target_label)
    
    if np.sum(is_clone) > 0:
        hits = np.cumsum(is_clone)
        prec = hits / (np.arange(len(is_clone)) + 1)
        aps.append(np.sum(prec * is_clone) / np.sum(is_clone))

print(f"\n✅ SCORE FINAL MAP : {np.mean(aps):.4f}")