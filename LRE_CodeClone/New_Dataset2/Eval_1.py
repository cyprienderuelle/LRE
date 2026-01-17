import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_ID = "naver/splade_v2_max"
LORA_PATH = "./checkpoint_epoch_1" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # Baissé un peu pour être sûr de la VRAM

print("🚀 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
model.eval()

# Chargement de la version officielle
print("📦 Chargement de CodeXGLUE POJ-104...")
dataset = load_dataset("microsoft/codexglue_code_to_code_clone_detection", split="test")

def get_embeddings(texts):
    embeddings = []
    # On limite à 1000 exemples pour un premier test rapide si besoin
    # texts = texts[:1000] 
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encodage"):
        batch = [str(t) for t in texts[i : i + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # Vecteur SPLADE
            vecs = torch.max(torch.log1p(torch.relu(outputs.logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
            embeddings.append(vecs.cpu())
    return torch.cat(embeddings)

# Dans microsoft/codexglue, les colonnes sont 'code' et 'label'
all_codes = dataset['code']
all_labels = np.array(dataset['label'])

print(f"🧬 Encodage de {len(all_codes)} programmes...")
all_vecs = get_embeddings(all_codes)

# --- CALCUL DU MAP ---
def calculate_map(embeddings, labels):
    # On utilise le produit scalaire pour SPLADE
    sim_matrix = torch.mm(embeddings, embeddings.t())
    sim_matrix.fill_diagonal_(-1e9)
    
    avg_precisions = []
    for i in tqdm(range(len(labels)), desc="Calcul MAP"):
        label = labels[i]
        # On récupère les scores pour la ligne i
        scores = sim_matrix[i]
        sorted_indices = torch.argsort(scores, descending=True).numpy()
        
        # Est-ce que le résultat i est du même label ?
        is_clone = (labels[sorted_indices] == label)
        
        if np.sum(is_clone) == 0: continue
        
        # Calcul Precision au rang k
        hits = np.cumsum(is_clone)
        precisions = hits / (np.arange(len(is_clone)) + 1)
        ap = np.sum(precisions * is_clone) / np.sum(is_clone)
        avg_precisions.append(ap)
        
    return np.mean(avg_precisions)

score = calculate_map(all_vecs, all_labels)
print(f"\n📊 MAP sur POJ-104: {score:.4f}")