import json
import torch
import random
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
model_id = "naver/splade_v2_max"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
model.eval()

def get_splade_embeddings(texts, batch_size=64):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding SPLADE"):
        batch = [t[:1000] for t in texts[i : i + batch_size]] # Troncature légère pour la rapidité
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            vecs = torch.max(torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1), dim=1).values
        
        # On normalise immédiatement pour accélérer le calcul cos_sim plus tard
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        all_embeddings.append(vecs.cpu())
    
    return torch.cat(all_embeddings)

# --- CHARGEMENT ---
input_file = 'Resultat.jsonl'
output_file = 'Dataset_InfoNCE_HardNeg.jsonl'

print("Chargement des données en mémoire...")
raw_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        raw_data.append(json.loads(line))

ancres_text = [d['anchor'] for d in raw_data]
ancres_text = ancres_text[:1000] # Pour test rapide, à enlever pour le dataset complet

# --- ENCODAGE ---
# Attention : si 500k, cela prendra ~2-4h selon le GPU
embeddings = get_splade_embeddings(ancres_text, batch_size=128)

# --- RECHERCHE ET SAUVEGARDE ---
print(f"Génération du fichier final : {output_file}")
embeddings = embeddings.to(device)

with open(output_file, 'w', encoding='utf-8') as f_out:
    # On traite par blocs pour la matrice de similarité (évite OOM GPU)
    step = 1000 
    for i in tqdm(range(0, len(ancres_text)), desc="Recherche Hard Negatives"):
        # Calcul des scores pour l'ancre actuelle contre TOUTES les autres
        query_vec = embeddings[i].unsqueeze(0)
        scores = torch.mm(query_vec, embeddings.t()).squeeze(0)
        
        # On récupère les top_k (k=10 pour être large et filtrer l'ancre elle-même)
        top_k = 10
        top_indices = torch.topk(scores, k=top_k).indices.cpu().tolist()
        
        # Filtrer l'ancre elle-même et garder le top 5
        neighbors = [idx for idx in top_indices if idx != i][:5]
        
        if not neighbors:
            # Sécurité si aucun voisin n'est trouvé (cas rare)
            neg_idx = random.randint(0, len(ancres_text)-1)
        else:
            neg_idx = random.choice(neighbors)
            
        # Construction de l'objet final
        output_obj = {
            "anchor": raw_data[i]['anchor'],
            "pos": raw_data[i]['positif'],
            "neg": raw_data[neg_idx]['anchor'],
            "type": raw_data[i].get('type', -1)
        }
        
        f_out.write(json.dumps(output_obj) + '\n')

print("Terminé !")