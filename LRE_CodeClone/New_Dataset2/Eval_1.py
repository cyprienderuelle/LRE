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

print("🚀 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
model.eval()

# Chargement de CodeSearchNet (C) - Très stable sur le Hub
print("📦 Chargement de CodeSearchNet (C)...")
dataset = load_dataset("code_search_net", "c", split="test", trust_remote_code=True)

# On prend un échantillon pour que l'évaluation soit rapide (ex: 2000 fonctions)
sample_size = min(2000, len(dataset))
all_codes = dataset['func_code_string'][:sample_size]
all_labels = dataset['func_name'][:sample_size]

def get_embeddings(texts):
    embeddings = []
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Encodage"):
        batch = [str(t)[:1000] for t in texts[i : i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            vecs = torch.max(torch.log1p(torch.relu(outputs.logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
            embeddings.append(vecs.cpu())
    return torch.cat(embeddings)

print(f"🧬 Encodage de {sample_size} fonctions C...")
all_vecs = get_embeddings(all_codes)

# --- CALCUL MAP ---
# (La logique reste la même que précédemment)
sim_matrix = torch.mm(all_vecs, all_vecs.t())
sim_matrix.fill_diagonal_(-1e9)

aps = []
for i in range(len(all_labels)):
    target_label = all_labels[i]
    scores = sim_matrix[i]
    sorted_indices = torch.argsort(scores, descending=True).numpy()
    
    # Ici on considère un "clone" si le nom de fonction est identique 
    # (Souvent vrai dans CodeSearchNet pour des versions différentes d'une lib)
    is_clone = (np.array(all_labels)[sorted_indices] == target_label)
    
    if np.sum(is_clone) > 0:
        hits = np.cumsum(is_clone)
        prec = hits / (np.arange(len(is_clone)) + 1)
        aps.append(np.sum(prec * is_clone) / np.sum(is_clone))

if aps:
    print(f"\n📊 MAP Score sur CodeSearchNet (C): {np.mean(aps):.4f}")
else:
    print("\n❌ Pas assez de clones trouvés pour calculer un score significatif.")