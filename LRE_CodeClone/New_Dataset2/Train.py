import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import json
from torch.utils.data import random_split
from peft import LoraConfig, get_peft_model

def load_triplets_optimized(filepath, max_size=1000000):

    print(f"Chargement des triplets depuis {filepath}")
    triplets = []
    with open(filepath, 'r', buffering=8192*16) as f:
        for i, line in enumerate(f):
            if i >= max_size:
                break
            data = json.loads(line)
            triplets.append((data["ancre"], data["pos"], data["neg"], data["type"]))

    return triplets

my_triplets_list = load_triplets_optimized("Dataset_InfoNCE_HardNeg.jsonl")

tmp_my_triplets_list = []
my_triplets_list_test = []

for i in range(len(my_triplets_list)):
  if i % 10 < 8:
    tmp_my_triplets_list.append((my_triplets_list[i][0], my_triplets_list[i][1], my_triplets_list[i][2], my_triplets_list[i][3]))
  else:
    my_triplets_list_test.append(my_triplets_list[i])

tmp_my_triplets_list = tmp_my_triplets_list

checkpoint = "naver/splade_v2_max"
batch_size = 128 # a augmenté
epochs = 5
lr = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_dim = 30522

# Dataset
class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_length=128):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative, _ = self.triplets[idx]
        a_enc = self.tokenizer(anchor, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        p_enc = self.tokenizer(positive, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        n_enc = self.tokenizer(negative, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        return a_enc, p_enc, n_enc

class SpladeTripletModel(nn.Module):
    def __init__(self, base_model, vocab_size=30522):
        super().__init__()
        self.base = base_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(base_model.config.hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        logits = self.proj(hidden)
        sparse_emb = torch.log1p(F.relu(logits))

        if attention_mask is not None:
            sparse_emb = sparse_emb * attention_mask.unsqueeze(-1)
        sparse_vec = torch.max(sparse_emb, dim=1).values
        return sparse_vec

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Paramètre de lissage (tau). 
                         Une valeur basse (0.07) rend le modèle plus sélectif.
        """
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, emb_anchor, emb_pos, emb_neg=None):
        """
        Args:
            emb_anchor: [batch_size, dim]
            emb_pos:    [batch_size, dim]
            emb_neg:    [batch_size, dim] (Optionnel si tu utilises les In-batch negatives)
        """
        emb_anchor = F.normalize(emb_anchor, p=2, dim=-1)
        emb_pos = F.normalize(emb_pos, p=2, dim=-1)
        logits = torch.matmul(emb_anchor, emb_pos.T) / self.temperature

        if emb_neg is not None:
            emb_neg = F.normalize(emb_neg, p=2, dim=-1)
            score_neg = torch.sum(emb_anchor * emb_neg, dim=-1, keepdim=True) / self.temperature
            logits = torch.cat([logits, score_neg], dim=1)

        labels = torch.arange(emb_anchor.size(0)).to(emb_anchor.device)

        return self.cross_entropy(logits, labels)

def compute_validation_metrics(model, dataloader, device, k=5):
    model.eval()
    
    correct_top1 = 0
    correct_topk = 0
    mrr_sum = 0
    total_dist_pos = 0
    total_dist_neg = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            a_enc, p_enc, n_enc = batch
            
            # 1. Encodage et Normalisation L2 (essentiel pour la similarité cosinus)
            emb_a = F.normalize(model(a_enc['input_ids'].squeeze(1).to(device), a_enc['attention_mask'].squeeze(1).to(device)), p=2, dim=-1)
            emb_p = F.normalize(model(p_enc['input_ids'].squeeze(1).to(device), p_enc['attention_mask'].squeeze(1).to(device)), p=2, dim=-1)
            emb_n = F.normalize(model(n_enc['input_ids'].squeeze(1).to(device), n_enc['attention_mask'].squeeze(1).to(device)), p=2, dim=-1)

            batch_size = emb_a.size(0)
            total_samples += batch_size

            # 2. Calcul des distances moyennes (L2) pour diagnostic
            total_dist_pos += torch.norm(emb_a - emb_p, p=2, dim=1).sum().item()
            total_dist_neg += torch.norm(emb_a - emb_n, p=2, dim=1).sum().item()

            # 3. Matrice de scores (Cosine Similarity via dot product sur vecteurs normalisés)
            # scores[i][j] = similarité entre ancre i et positif j
            scores = torch.matmul(emb_a, emb_p.T) 
            target = torch.arange(batch_size).to(device)

            # --- TOP-1 & TOP-K ---
            _, topk_indices = scores.topk(k, dim=1)
            correct_top1 += (topk_indices[:, 0] == target).sum().item()
            correct_topk += (topk_indices == target.unsqueeze(1)).any(dim=1).sum().item()

            # --- MEAN RECIPROCAL RANK (MRR) ---
            # On trie tous les scores par ligne
            _, all_indices = torch.sort(scores, descending=True, dim=1)
            
            # On trouve le rang du vrai positif pour chaque ancre du batch
            # (target == all_indices).nonzero() renvoie les coordonnées du 1.0
            ranks = (all_indices == target.unsqueeze(1)).nonzero()[:, 1] + 1
            mrr_sum += (1.0 / ranks.float()).sum().item()

    # Calcul des moyennes finales
    results = {
        "top1_acc": (correct_top1 / total_samples) * 100,
        f"top{k}_acc": (correct_topk / total_samples) * 100,
        "mrr": mrr_sum / total_samples,
        "dist_pos_avg": total_dist_pos / total_samples,
        "dist_neg_avg": total_dist_neg / total_samples,
        "contrast_gap": (total_dist_neg - total_dist_pos) / total_samples
    }

    return results

triplets_dataset = TripletDataset(tmp_my_triplets_list, AutoTokenizer.from_pretrained(checkpoint))

train_size = int(0.8 * len(triplets_dataset))
val_size = len(triplets_dataset) - train_size

train_dataset, val_dataset = random_split(triplets_dataset, [train_size, val_size])

# 4. Création des DataLoaders respectifs
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModel.from_pretrained(checkpoint)
model = SpladeTripletModel(base_model).to(device)
model.base.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # Noms exacts des couches Linéaires pour DistilBERT
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"], 
    lora_dropout=0.05,
    bias="none",
    # On garde 'proj' si tu as une couche personnalisée dans ton SpladeModel
    modules_to_save=["proj"] 
)

# 3. Appliquer LoRA
model.base = get_peft_model(base_model, lora_config)

# Afficher le nombre de paramètres entraînables
model.base.print_trainable_parameters()

loss_fn = InfoNCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ----------------------------
# Training loop
# ----------------------------
base_save_path = "."

training_log_path = base_save_path + "/training_history.json"
training_history = []

for epoch in range(epochs):
    model.train()

    # Métrics
    total_loss = 0
    total_triplet_loss = 0
    total_sparsity_loss = 0
    batch_losses = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_idx, batch in enumerate(pbar):
        a_enc, p_enc, n_enc = batch

        a_ids = a_enc['input_ids'].squeeze(1).to(device)
        a_mask = a_enc['attention_mask'].squeeze(1).to(device)
        p_ids = p_enc['input_ids'].squeeze(1).to(device)
        p_mask = p_enc['attention_mask'].squeeze(1).to(device)
        n_ids = n_enc['input_ids'].squeeze(1).to(device)
        n_mask = n_enc['attention_mask'].squeeze(1).to(device)

        optimizer.zero_grad()
        emb_a = model(a_ids, a_mask)
        emb_p = model(p_ids, p_mask)
        emb_n = model(n_ids, n_mask)

        loss = loss_fn(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer.step()

        # Accumulation des métriques
        batch_loss = loss.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)

        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })

    pbar.close()

    # SAUVEGARDE DU L'EPOCH
    model.eval()

    total_val_loss = 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
    
    with torch.no_grad(): # Indispensable pour ne pas saturer la VRAM
        for batch in val_pbar:
            a_enc, p_enc, n_enc = batch
            
            a_ids = a_enc['input_ids'].squeeze(1).to(device)
            a_mask = a_enc['attention_mask'].squeeze(1).to(device)
            p_ids = p_enc['input_ids'].squeeze(1).to(device)
            p_mask = p_enc['attention_mask'].squeeze(1).to(device)
            n_ids = n_enc['input_ids'].squeeze(1).to(device)
            n_mask = n_enc['attention_mask'].squeeze(1).to(device)

            emb_a = model(a_ids, a_mask)
            emb_p = model(p_ids, p_mask)
            emb_n = model(n_ids, n_mask)

            v_loss = loss_fn(emb_a, emb_p, emb_n)
            total_val_loss += v_loss.item()
            
            val_pbar.set_postfix({'val_loss': f'{v_loss.item():.4f}'})

    avg_val_loss = total_val_loss / len(val_loader)

    # --- Dans ta boucle d'époque, après la validation ---
    res = compute_validation_metrics(model, val_loader, device, k=5)

    print(f"Validation Accuracy: Top-1: {res['top1_acc']:.2f}% | Top-5: {res['top5_acc']:.2f}%")

    avg_loss = total_loss / len(train_loader)
    avg_triplet_loss = total_triplet_loss / len(train_loader)
    avg_sparsity_loss = total_sparsity_loss / len(train_loader)

    # Calcul de la sparsité moyenne
    with torch.no_grad():
        # Test sur un batch pour mesurer la sparsité
        sample_text = "exemple de texte"
        sample_enc = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
        sample_emb = model(sample_enc['input_ids'].to(device), sample_enc['attention_mask'].to(device))
        non_zero = (sample_emb.abs() > 0.01).sum().item()
        total_dims = sample_emb.numel()
        sparsity_percent = 100 * (1 - non_zero / total_dims)

    # Sauvegarde des informations de l'époque
    epoch_info = {
        "epoch": epoch + 1,
        "losses": {
            "total": avg_loss,
            "triplet": avg_triplet_loss,
            "sparsity": avg_sparsity_loss,
            "val_avg": avg_val_loss,
            "min_batch": min(batch_losses),
            "max_batch": max(batch_losses)
        },
        "metrics": {
            "sparsity_percent": sparsity_percent,
            "non_zero_dims": non_zero,
            "top1_acc": res['top1_acc'],
            "top5_acc": res['top5_acc'],
            "mmr": res['mrr'],
            "dist_pos_avg": res['dist_pos_avg'],
            "dist_neg_avg": res['dist_neg_avg'],
            "contrast_gap": res['contrast_gap'],
            "total_dims": total_dims
        },
        "hyperparameters": {
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": batch_size,
            "num_batches": len(train_loader)
        }
    }

    training_history.append(epoch_info)

    checkpoint_dir = os.path.join(base_save_path, f"checkpoint_epoch_{epoch+1}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.base.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    torch.save({
        'projection_layer': model.proj.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch + 1,
        'loss': avg_loss,
    }, os.path.join(checkpoint_dir, "training_state.pt"))

    # Sauvegarder les infos de l'époque
    with open(os.path.join(checkpoint_dir, "epoch_info.json"), 'w') as f:
        json.dump(epoch_info, f, indent=2)

    # Mettre à jour l'historique complet
    with open(training_log_path, 'w') as f:
        json.dump({
            "total_epochs": epochs,
            "model_config": {
                "checkpoint": checkpoint,
                "batch_size": batch_size,
                "learning_rate": lr
            },
            "history": training_history
        }, f, indent=2)

# ----------------------------
# Sauvegarde du modèle final
# ----------------------------
final_model_path = os.path.join(base_save_path, "final_model")
os.makedirs(final_model_path, exist_ok=True)

model.base.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
torch.save({
    'projection_layer': model.proj.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'epoch': epochs,
    'final_loss': avg_loss,
}, os.path.join(final_model_path, "training_state.pt"))

print(f"\n{'='*70}")
print(f"✅ Entraînement terminé!")
print(f"{'='*70}")
print(f"Historique complet: {training_log_path}")
