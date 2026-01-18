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
LORA_LIST = ["./checkpoint_epoch_v2_1", "./checkpoint_epoch_v2_2", "./checkpoint_epoch_v2_3", "./checkpoint_epoch_81acc_1", "./checkpoint_epoch_map70"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
SAMPLE_SIZE = 1000 # On réduit car on compare Query vs Code (matrice de similarité)

def run_codesearchnet_eval(model, tokenizer, queries, codes, desc="Evaluation"):
    model.eval()
    
    def get_embeddings(texts, type_desc):
        embeddings = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Encodage {type_desc} ({desc})"):
            batch = [str(t)[:1000] for t in texts[i : i + BATCH_SIZE]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                vecs = model(inputs['input_ids'], inputs['attention_mask'])
                embeddings.append(vecs.cpu())
        return torch.cat(embeddings)

    # On encode les docstrings (requêtes) et les fonctions (codes) séparément
    query_vecs = get_embeddings(queries, "Queries")
    code_vecs = get_embeddings(codes, "Codes")
    
    print(f"📊 Calcul MRR pour {desc}...")
    # Similarité entre chaque requête et chaque code
    sim_matrix = torch.mm(query_vecs, code_vecs.t()) 
    
    ranks = []
    for i in range(len(queries)):
        # Score de la paire correcte (diagonale : query i avec code i)
        correct_score = sim_matrix[i, i]
        # On compte combien de codes ont un score supérieur au code correct
        rank = (sim_matrix[i] > correct_score).sum().item() + 1
        ranks.append(1.0 / rank)
    
    return np.mean(ranks)

# 1. Chargement de CodeSearchNet (Exemple: Python)
print("📦 Chargement de CodeSearchNet (C)...")
# Note: Tu peux essayer "java", "go", "javascript", etc.
dataset = load_dataset("code-search-net/code_search_net", "cpp", split="test", trust_remote_code=False)

# On récupère les colonnes spécifiques : docstring (query) et code (target)
all_queries = dataset['func_documentation_string'][:SAMPLE_SIZE]
all_codes = dataset['func_code_string'][:SAMPLE_SIZE]

# 2. TEST BASELINE
print("\n--- TEST 1: BASELINE (Modèle original) ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_mlm = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
model_baseline = SpladeTripletModel(base_mlm).to(DEVICE)

mrr_baseline = run_codesearchnet_eval(model_baseline, tokenizer, all_queries, all_codes, desc="Baseline")
print(f"❌ MRR Baseline: {mrr_baseline:.4f}")

del model_baseline, base_mlm
torch.cuda.empty_cache()

# 3. TEST CHECKPOINTS LORA
results = {"Baseline": mrr_baseline}

for LORA_PATH in LORA_LIST:
    print(f"\n--- TEST: {LORA_PATH} ---")
    base_mlm_lora = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    model_peft = PeftModel.from_pretrained(base_mlm_lora, LORA_PATH)
    model_lora = SpladeTripletModel(model_peft).to(DEVICE)

    mrr_lora = run_codesearchnet_eval(model_lora, tokenizer, all_queries, all_codes, desc="LoRA")
    results[LORA_PATH] = mrr_lora
    print(f"✅ MRR LoRA: {mrr_lora:.4f}")
    
    del model_lora, base_mlm_lora
    torch.cuda.empty_cache()

# --- RÉSUMÉ ---
print("\n" + "="*40)
print(f"{'Modèle':<25} | {'MRR':<10}")
print("-" * 40)
for name, mrr in results.items():
    print(f"{name:<25} | {mrr:.4f}")
print("="*40)