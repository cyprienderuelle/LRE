import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from Train import SpladeTripletModel

path_dataset_test = "./new_piscine_dataset.jsonl"
base_path = "."
device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger et grouper les triplets par type d'ancre
triplets_by_type = defaultdict(list)

with open(path_dataset_test, "r") as f:
    for line in f:
        data = json.loads(line)
        triplets_by_type[data["function_type"]].append({
            "ancre": data["anchor"],
            "pos": data["positive"],
            "neg": data["negative"],
            "type": data["function_type"]
        })

print(f"Dataset d'évaluation: {sum(len(v) for v in triplets_by_type.values())} triplets")
print(f"Nombre de types différents: {len(triplets_by_type)}\n")

# Dans evaluation.py, modifie la fonction load_model_from_checkpoint
def load_model_from_checkpoint(checkpoint_path, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    # 1. Utilise ForMaskedLM au lieu de AutoModel
    base_model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
    model = SpladeTripletModel(base_model).to(device)

    model.eval()
    return model, tokenizer

@torch.no_grad()
def encode_batch(texts, model, tokenizer, device, batch_size=8):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=128)

        input_ids = enc['input_ids'].to(device, non_blocking=True)
        attention_mask = enc['attention_mask'].to(device, non_blocking=True)

        embeddings = model(input_ids, attention_mask)
        embeddings = embeddings.cpu()

        all_embeddings.append(embeddings)

        del input_ids, attention_mask, enc, embeddings
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    result = torch.cat(all_embeddings, dim=0)
    del all_embeddings

    return result

def compute_ndcg_at_k(rankings, k):
    dcg = 0.0
    for rank in rankings:
        if rank <= k:
            dcg += 1.0 / np.log2(rank + 1)

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rankings), k)))

    return dcg / idcg if idcg > 0 else 0.0

@torch.no_grad()
def compute_metrics_with_candidates(triplets_by_type, model, tokenizer, device, batch_size=4):
    all_accuracies = []
    all_ndcg_at_1 = []
    all_ndcg_at_3 = []
    all_ndcg_at_5 = []
    all_ndcg_at_10 = []
    all_ndcg_at_20 = []
    all_mrr = []

    # Extraire toutes les ancres uniques
    all_anchors = []
    anchor_to_types = defaultdict(list)  # ancre -> [liste des types]

    for type_ancre, triplets in triplets_by_type.items():
        for t in triplets:
            anchor_text = t["ancre"]
            if anchor_text not in [a["text"] for a in all_anchors]:
                all_anchors.append({"text": anchor_text, "types": []})

            # Associer cette ancre à son type et ses pos/neg
            for anchor in all_anchors:
                if anchor["text"] == anchor_text:
                    anchor["types"].append({
                        "type": type_ancre,
                        "pos": t["pos"],
                        "neg": t["neg"]
                    })
                    break

    print(f"\n📊 Total: {len(all_anchors)} ancres uniques avec {len(triplets_by_type)} types différents")

    # Pour chaque type, préparer les candidats
    candidates_by_type = {}
    embeddings_by_type = {}

    for type_ancre, triplets in triplets_by_type.items():
        # Collecter tous les pos et neg de ce type
        all_pos_for_type = list(set([t["pos"] for t in triplets]))
        all_neg_for_type = list(set([t["neg"] for t in triplets]))

        # Créer la liste de tous les candidats pour ce type
        all_candidates = all_pos_for_type + all_neg_for_type
        all_candidate_labels = [1] * len(all_pos_for_type) + [0] * len(all_neg_for_type)

        candidates_by_type[type_ancre] = {
            "texts": all_candidates,
            "labels": all_candidate_labels
        }

        # Encoder tous les candidats une seule fois pour ce type
        embeddings_by_type[type_ancre] = encode_batch(all_candidates, model, tokenizer, device, batch_size=8).to(device)

    print(f"\n🔄 Évaluation des {len(all_anchors)} ancres sur tous leurs types...")

    # Évaluer chaque ancre
    pbar = tqdm(total=len(all_anchors), desc="Evaluation", ncols=100)

    for i in range(0, len(all_anchors), batch_size):
        batch_anchors = all_anchors[i:i+batch_size]
        anchor_texts = [a["text"] for a in batch_anchors]

        # Encoder les ancres
        emb_anchors = encode_batch(anchor_texts, model, tokenizer, device, batch_size=batch_size).to(device)

        # Pour chaque ancre du batch
        for anchor_idx, anchor_data in enumerate(batch_anchors):
            anchor_emb = emb_anchors[anchor_idx:anchor_idx+1]

            # Évaluer cette ancre sur chacun de ses types
            for type_info in anchor_data["types"]:
                type_name = type_info["type"]

                # Récupérer les candidats et embeddings pour ce type
                emb_candidates = embeddings_by_type[type_name]
                candidate_labels = candidates_by_type[type_name]["labels"]

                scores = torch.sum(anchor_emb * emb_candidates, dim=1)

                # Trier par score décroissant
                sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
                sorted_labels = [candidate_labels[idx] for idx in sorted_indices]

                # Accuracy: le premier est-il positif?
                all_accuracies.append(1.0 if sorted_labels[0] == 1 else 0.0)

                # MRR: rang du premier positif
                try:
                    first_pos_rank = sorted_labels.index(1) + 1
                    all_mrr.append(1.0 / first_pos_rank)
                except ValueError:
                    all_mrr.append(0.0)

                # NDCG: trouver les rangs de tous les positifs
                positive_ranks = [idx + 1 for idx, label in enumerate(sorted_labels) if label == 1]

                all_ndcg_at_1.append(compute_ndcg_at_k(positive_ranks, 1))
                all_ndcg_at_3.append(compute_ndcg_at_k(positive_ranks, 3))
                all_ndcg_at_5.append(compute_ndcg_at_k(positive_ranks, 5))
                all_ndcg_at_10.append(compute_ndcg_at_k(positive_ranks, 10))
                all_ndcg_at_20.append(compute_ndcg_at_k(positive_ranks, 20))

        pbar.update(len(batch_anchors))

        # Nettoyage
        del emb_anchors
        torch.cuda.empty_cache()

    pbar.close()

    # Nettoyer les embeddings des candidats
    for emb in embeddings_by_type.values():
        del emb
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    metrics = {
        'accuracy': np.mean(all_accuracies),
        'mrr': np.mean(all_mrr),
        'ndcg@1': np.mean(all_ndcg_at_1),
        'ndcg@3': np.mean(all_ndcg_at_3),
        'ndcg@5': np.mean(all_ndcg_at_5),
        'ndcg@10': np.mean(all_ndcg_at_10),
        'ndcg@20': np.mean(all_ndcg_at_20)
    }

    return metrics


# Trouver tous les checkpoints
base_path_obj = Path(base_path)
checkpoints = sorted([d for d in base_path_obj.iterdir()
                     if d.is_dir() and d.name.startswith('checkpoint_epoch_1')],
                    key=lambda x: int(x.name.split('_')[-1]))

print(f"Checkpoints trouvés: {len(checkpoints)}\n")

# Évaluation de tous les checkpoints
results = []

for checkpoint_dir in checkpoints:
    epoch_num = int(checkpoint_dir.name.split('_')[-1])

    model, tokenizer = load_model_from_checkpoint(str(checkpoint_dir), device)

    metrics = compute_metrics_with_candidates(triplets_by_type, model, tokenizer, device, batch_size=4)

    result = {
        'epoch': epoch_num,
        'checkpoint_path': str(checkpoint_dir),
        **metrics
    }
    results.append(result)

    print(f"Evaluating checkpoint {checkpoint_dir}")
    print(f"  ✓ Accuracy:       {metrics['accuracy']*100:.2f}%")
    print(f"  ✓ MRR:            {metrics['mrr']:.4f}")
    print(f"  ✓ NDCG@1:         {metrics['ndcg@1']:.4f}")
    print(f"  ✓ NDCG@3:         {metrics['ndcg@3']:.4f}")
    print(f"  ✓ NDCG@5:         {metrics['ndcg@5']:.4f}")
    print(f"  ✓ NDCG@10:        {metrics['ndcg@10']:.4f}")
    print(f"  ✓ NDCG@20:        {metrics['ndcg@20']:.4f}")
    print()

    del model
    torch.cuda.empty_cache()

# Sauvegarder les résultats
results_path = base_path_obj / "evaluation_results22222.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"💾 Résultats sauvegardés: {results_path}\n")

# Visualisation et graphe
epochs = [r['epoch'] for r in results]
accuracies = [r['accuracy'] * 100 for r in results]
mrr_scores = [r['mrr'] for r in results]
ndcg1_scores = [r['ndcg@1'] for r in results]
ndcg3_scores = [r['ndcg@3'] for r in results]
ndcg5_scores = [r['ndcg@5'] for r in results]
ndcg10_scores = [r['ndcg@10'] for r in results]
ndcg20_scores = [r['ndcg@20'] for r in results]

plt.figure(figsize=(18, 12))

# Subplot 1: Accuracy
plt.subplot(3, 3, 1)
plt.plot(epochs, accuracies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('Accuracy par Epoch', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)

# Subplot 2: MRR
plt.subplot(3, 3, 2)
plt.plot(epochs, mrr_scores, 's-', linewidth=2, markersize=8, color='#A23B72')
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('MRR', fontsize=11, fontweight='bold')
plt.title('MRR par Epoch', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])

# Subplot 3: NDCG@1
plt.subplot(3, 3, 3)
plt.plot(epochs, ndcg1_scores, '^-', linewidth=2, markersize=8, color='#F18F01')
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('NDCG@1', fontsize=11, fontweight='bold')
plt.title('NDCG@1 par Epoch', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])

# Subplot 4: NDCG@3
plt.subplot(3, 3, 4)
plt.plot(epochs, ndcg3_scores, 'd-', linewidth=2, markersize=8, color='#6A994E')
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('NDCG@3', fontsize=11, fontweight='bold')
plt.title('NDCG@3 par Epoch', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])

# Subplot 5: NDCG@5
plt.subplot(3, 3, 5)
plt.plot(epochs, ndcg5_scores, 'v-', linewidth=2, markersize=8, color='#BC4749')
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('NDCG@5', fontsize=11, fontweight='bold')
plt.title('NDCG@5 par Epoch', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])

# Subplot 6: NDCG@10
plt.subplot(3, 3, 6)
plt.plot(epochs, ndcg10_scores, 'p-', linewidth=2, markersize=8, color='#9B59B6')
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('NDCG@10', fontsize=11, fontweight='bold')
plt.title('NDCG@10 par Epoch', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])

# Subplot 7: NDCG@20
plt.subplot(3, 3, 7)
plt.plot(epochs, ndcg20_scores, 'h-', linewidth=2, markersize=8, color='#E63946')
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('NDCG@20', fontsize=11, fontweight='bold')
plt.title('NDCG@20 par Epoch', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])

# Subplot 8: Comparaison NDCG (tous)
plt.subplot(3, 3, 8)
plt.plot(epochs, ndcg1_scores, 'o-', linewidth=2, label='NDCG@1', color='#F18F01')
plt.plot(epochs, ndcg3_scores, 's-', linewidth=2, label='NDCG@3', color='#6A994E')
plt.plot(epochs, ndcg5_scores, '^-', linewidth=2, label='NDCG@5', color='#BC4749')
plt.plot(epochs, ndcg10_scores, 'v-', linewidth=2, label='NDCG@10', color='#9B59B6')
plt.plot(epochs, ndcg20_scores, 'd-', linewidth=2, label='NDCG@20', color='#E63946')
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('NDCG Score', fontsize=11, fontweight='bold')
plt.title('Comparaison NDCG', fontsize=13, fontweight='bold')
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])

# Subplot 9: Vue d'ensemble (Accuracy + MRR)
plt.subplot(3, 3, 9)
ax1 = plt.gca()
ax1.plot(epochs, accuracies, 'o-', linewidth=2, label='Accuracy (%)', color='#2E86AB')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#2E86AB')
ax1.tick_params(axis='y', labelcolor='#2E86AB')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(epochs, mrr_scores, 's-', linewidth=2, label='MRR', color='#A23B72')
ax2.set_ylabel('MRR', fontsize=11, fontweight='bold', color='#A23B72')
ax2.tick_params(axis='y', labelcolor='#A23B72')
ax2.set_ylim([0, 1.05])

plt.title('Accuracy + MRR', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(base_path_obj / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: evaluation_metrics.png")
plt.show()