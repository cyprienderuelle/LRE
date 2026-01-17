import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_ID = "naver/splade_v2_max"
LORA_PATH = "./checkpoint_epoch_81acc_1" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 

print("🚀 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
model.eval()

# Tentative de chargement avec le nouveau chemin
print("📦 Chargement de CodeXGLUE POJ-104...")
try:
    dataset = load_dataset("code_x_glue/code_to_code_clone_detection", split="test")
except Exception as e:
    print(f"Erreur de chargement : {e}")
    print("Tentative avec un autre chemin...")
    dataset = load_dataset("code_x_glue/code_to_code_clone_detection", "default", split="test")

def get_embeddings(texts):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encodage SPLADE"):
        batch = [str(t) for t in texts[i : i + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # Logique SPLADE : max pooling sur log(1 + ReLU(logits))
            vecs = torch.max(torch.log1p(torch.relu(outputs.logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
            embeddings.append(vecs.cpu())
    return torch.cat(embeddings)

# Dans CodeXGLUE POJ-104, les colonnes sont généralement 'code' et 'label' (ou 'id')
all_codes = dataset['code']
# On vérifie si 'label' existe, sinon on prend 'id'
label_col = 'label' if 'label' in dataset.column_names else 'id'
all_labels = np.array(dataset[label_col])

print(f"🧬 Encodage de {len(all_codes)} programmes...")
all_vecs = get_embeddings(all_codes)

# --- CALCUL DU MAP ---
def calculate_map(embeddings, labels):
    # Similiarité par produit scalaire pour SPLADE
    sim_matrix = torch.mm(embeddings, embeddings.t())
    # On masque la diagonale (soi-même)
    sim_matrix.fill_diagonal_(-1e9)
    
    avg_precisions = []
    for i in tqdm(range(len(labels)), desc="Calcul MAP"):
        label = labels[i]
        scores = sim_matrix[i]
        sorted_indices = torch.argsort(scores, descending=True).numpy()
        
        # Un clone est un code avec le même label/id
        is_clone = (labels[sorted_indices] == label)
        
        if np.sum(is_clone) == 0: continue
        
        # Precision @ K
        hits = np.cumsum(is_clone)
        precisions = hits / (np.arange(len(is_clone)) + 1)
        ap = np.sum(precisions * is_clone) / np.sum(is_clone)
        avg_precisions.append(ap)
        
    return np.mean(avg_precisions)

score = calculate_map(all_vecs, all_labels)
print(f"\n" + "="*35)
print(f"✅ RÉSULTAT FINAL POJ-104")
print(f"MAP Score : {score:.4f}")
print(f"="*35)