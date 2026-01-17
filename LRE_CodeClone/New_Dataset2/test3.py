import json
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel

# --- CONFIG ---
model_id = "naver/splade_v2_max"
lora_path = "./checkpoint_epoch_1" # Ton modèle à 81%
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base = AutoModelForMaskedLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base, lora_path).to(device)
model.eval()

def get_splade_vec(text):
    with torch.no_grad():
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        out = model(**tokens)
        vec = torch.max(torch.log1p(torch.relu(out.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1).values
        return vec

# --- COMPARAISON ---
file1 = 'Dataset_InfoNCE_HardNeg.jsonl'
file2 = 'Dataset_InfoNCE_HardNeg_V2.jsonl'

similarities_v1 = []
similarities_v2 = []

with open(file1, 'r') as f1, open(file2, 'r') as f2:
    # On compare les 100 premiers pour avoir une idée
    for _ in range(100):
        d1 = json.loads(f1.readline())
        d2 = json.loads(f2.readline())
        
        # On encode l'ancre et les deux négatifs
        v_ancre = get_splade_vec(d1['ancre'])
        v_neg1 = get_splade_vec(d1['neg'])
        v_neg2 = get_splade_vec(d2['neg'])
        
        # Produit scalaire (Sim SPLADE)
        sim1 = torch.mm(v_ancre, v_neg1.t()).item()
        sim2 = torch.mm(v_ancre, v_neg2.t()).item()
        
        similarities_v1.append(sim1)
        similarities_v2.append(sim2)

print(f"Score de difficulté moyen (V1) : {sum(similarities_v1)/100:.2f}")
print(f"Score de difficulté moyen (V2) : {sum(similarities_v2)/100:.2f}")