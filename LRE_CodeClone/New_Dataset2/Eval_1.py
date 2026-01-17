import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# --- CONFIG ---
MODEL_ID = "naver/splade_v2_max"
LORA_PATH = "./checkpoint_epoch_81acc_1" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

print("🚀 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
model.eval()

# 1. Chargement du dataset standard (Format Parquet Garanti)
print("📦 Chargement de CodeXGLUE Clone Detection...")
try:
    # C'est le chemin le plus robuste actuellement
    dataset = load_dataset("code_x_glue_cc_clone_detection", split="test")
except Exception as e:
    print(f"Échec du chargement standard, tentative alternative...")
    # Au cas où, on utilise le chemin complet
    dataset = load_dataset("microsoft/codexglue_code_to_code_clone_detection", split="test", trust_remote_code=True)

# 2. Préparation des données
# Dans ce dataset, 'code' contient le code source et 'id' ou 'label' contient l'identifiant du problème
all_codes = dataset['code']
# On vérifie la colonne de label
label_col = 'label' if 'label' in dataset.column_names else 'id'
all_labels = np.array(dataset[label_col])

# On limite à 2000 pour un premier test rapide
sample_size = min(2000, len(all_codes))
all_codes = all_codes[:sample_size]
all_labels = all_labels[:sample_size]

def get_embeddings(texts):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encodage SPLADE"):
        batch = [str(t)[:1024] for t in texts[i : i + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # SPLADE pooling
            vecs = torch.max(torch.log1p(torch.relu(outputs.logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
            embeddings.append(vecs.cpu())
    return torch.cat(embeddings)

print(f"🧬 Encodage de {sample_size} programmes C...")
all_vecs = get_embeddings(all_codes)

# --- CALCUL DU MAP ---
print("📊 Calcul du score MAP...")
# Normalisation pour utiliser le produit scalaire comme une similarité cosinus si nécessaire
# Mais SPLADE fonctionne mieux avec le produit scalaire pur (Dot Product)
sim_matrix = torch.mm(all_vecs, all_vecs.t())
sim_matrix.fill_diagonal_(-1e9)

aps = []
for i in range(len(all_labels)):
    target_label = all_labels[i]
    scores = sim_matrix[i]
    sorted_indices = torch.argsort(scores, descending=True).numpy()
    
    # Un clone = même ID de problème
    is_clone = (all_labels[sorted_indices] == target_label)
    
    if np.sum(is_clone) > 0:
        hits = np.cumsum(is_clone)
        prec = hits / (np.arange(len(is_clone)) + 1)
        # Average Precision
        ap = np.sum(prec * is_clone) / np.sum(is_clone)
        aps.append(ap)

if aps:
    print(f"\n" + "="*40)
    print(f"🏆 RÉSULTAT ÉVALUATION CLONE DETECTION")
    print(f"MAP Score : {np.mean(aps):.4f}")
    print(f"="*40)
else:
    print("\n❌ Erreur : Aucun clone trouvé pour le calcul.")