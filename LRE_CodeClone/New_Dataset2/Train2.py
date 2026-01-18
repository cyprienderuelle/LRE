import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import os
import json
from torch.utils.data import random_split

# --- Classes Dataset et Model (Identiques) ---

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
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model 

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits 
        sparse_emb = torch.log1p(F.relu(logits))
        if attention_mask is not None:
            sparse_emb = sparse_emb * attention_mask.unsqueeze(-1)
        return torch.max(sparse_emb, dim=1).values

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, emb_anchor, emb_pos, emb_neg=None):
        # Pour plus de stabilité en Full Fine-Tuning, on peut normaliser :
        # emb_anchor = F.normalize(emb_anchor, p=2, dim=-1)
        # emb_pos = F.normalize(emb_pos, p=2, dim=-1)
        
        logits = torch.matmul(emb_anchor, emb_pos.T) / self.temperature
        if emb_neg is not None:
            # emb_neg = F.normalize(emb_neg, p=2, dim=-1)
            score_neg = torch.sum(emb_anchor * emb_neg, dim=-1, keepdim=True) / self.temperature
            logits = torch.cat([logits, score_neg], dim=1)

        labels = torch.arange(emb_anchor.size(0)).to(emb_anchor.device)
        return self.cross_entropy(logits, labels)

# --- Fonctions Utilitaires ---

def compute_validation_metrics(model, dataloader, device, k=5):
    model.eval()
    correct_top1, correct_topk, mrr_sum, total_samples = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            a_enc, p_enc, _ = batch
            emb_a = model(a_enc['input_ids'].squeeze(1).to(device), a_enc['attention_mask'].squeeze(1).to(device))
            emb_p = model(p_enc['input_ids'].squeeze(1).to(device), p_enc['attention_mask'].squeeze(1).to(device))

            batch_size = emb_a.size(0)
            total_samples += batch_size
            scores = torch.matmul(emb_a, emb_p.T) 
            target = torch.arange(batch_size).to(device)

            _, topk_indices = scores.topk(k, dim=1)
            correct_top1 += (topk_indices[:, 0] == target).sum().item()
            correct_topk += (topk_indices == target.unsqueeze(1)).any(dim=1).sum().item()
            _, all_indices = torch.sort(scores, descending=True, dim=1)
            ranks = (all_indices == target.unsqueeze(1)).nonzero()[:, 1] + 1
            mrr_sum += (1.0 / ranks.float()).sum().item()

    return {"top1_acc": (correct_top1/total_samples)*100, "topk_acc": (correct_topk/total_samples)*100, "mrr": mrr_sum/total_samples}

# --- Main Training Loop ---

if __name__ == "__main__":
    # Paramètres
    checkpoint = "naver/splade_v2_max"
    batch_size = 64 # RÉDUIT pour le Full Fine-Tuning (plus de mémoire nécessaire)
    epochs = 3
    lr = 1e-5 # Learning rate plus faible pour ne pas détruire les poids pré-entraînés
    lambda_sparsity = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Chargement Data
    def load_triplets(filepath, max_size=100000):
        triplets = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_size: break
                d = json.loads(line)
                triplets.append((d["ancre"], d["pos"], d["neg"], d["type"]))
        return triplets

    data_list = load_triplets("Dataset_InfoNCE_HardNeg.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = TripletDataset(data_list, tokenizer)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # 2. Modèle SANS LoRA
    print("🚀 Initialisation Full Fine-Tuning...")
    base_model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    model = SpladeTripletModel(base_model).to(device)
    
    # On s'assure que TOUT est entraînable
    for param in model.parameters():
        param.requires_grad = True
    
    model.base.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = InfoNCELoss()

    # 3. Boucle d'entraînement
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            a_ids = batch[0]['input_ids'].squeeze(1).to(device)
            a_mask = batch[0]['attention_mask'].squeeze(1).to(device)
            p_ids = batch[1]['input_ids'].squeeze(1).to(device)
            p_mask = batch[1]['attention_mask'].squeeze(1).to(device)
            n_ids = batch[2]['input_ids'].squeeze(1).to(device)
            n_mask = batch[2]['attention_mask'].squeeze(1).to(device)

            emb_a, emb_p, emb_n = model(a_ids, a_mask), model(p_ids, p_mask), model(n_ids, n_mask)
            
            loss_c = loss_fn(emb_a, emb_p, emb_n)
            l1 = (torch.mean(torch.sum(torch.abs(emb_a), dim=-1)) + 
                  torch.mean(torch.sum(torch.abs(emb_p), dim=-1)) + 
                  torch.mean(torch.sum(torch.abs(emb_n), dim=-1))) / 3
            
            loss = loss_c + (lambda_sparsity * l1)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        res = compute_validation_metrics(model, val_loader, device)
        print(f"✅ Epoch {epoch+1} | MRR: {res['mrr']:.4f} | Top-1: {res['top1_acc']:.2f}%")

        # Sauvegarde
        save_path = f"checkpoint_full_epoch_v3_{epoch+1}"
        model.base.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)