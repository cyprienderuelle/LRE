import os
import glob
import json
from tqdm import tqdm
from collections import defaultdict
import time

import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True

BATCH_SIZE = 32

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Temps d'exécution de {func.__name__}: {end_time - start_time:.2f} secondes")
        return result
    return wrapper

@measure_time
def load_data():
    with open("dataset_splade_mesure.jsonl") as f:
        data = [json.loads(line) for line in f]
    dataset_mesure = Dataset.from_list(data)
    print("Taille du dataset de mesure : ", len(dataset_mesure))
    return data, dataset_mesure

@measure_time
def extract_functions(data):
    base_functions = []
    base_functions_types = []
    base_functions_dict = {}
    modified_functions = []
    modified_functions_types = []

    for item in data:
        document = item["document"]
        document_type = item["type"]
        if document not in base_functions_dict:
            base_functions_dict[document] = document_type
            base_functions.append(document)
            base_functions_types.append(document_type)

    for item in data:
        if item["label"] == 1:
            modified_functions.append(item["query"])
            modified_functions_types.append(item["type"])

    print(f"Nombre de fonctions de base: {len(base_functions)}")
    print(f"Nombre de fonctions modifiées (label=1): {len(modified_functions)}")
    
    return base_functions, base_functions_types, modified_functions, modified_functions_types

@measure_time
def compute_similarity_matrix(model, tokenizer, base_functions, modified_functions):
    similarity_matrix = np.zeros((len(base_functions), len(modified_functions)))
    
    for i in tqdm(range(0, len(base_functions), BATCH_SIZE), desc="Calcul des similarités", leave=False):
        batch_base = base_functions[i:i+BATCH_SIZE]
        
        for j in range(0, len(modified_functions), BATCH_SIZE):
            batch_modified = modified_functions[j:j+BATCH_SIZE]
            
            pairs = []
            pair_indices = []
            
            for base_idx, base_func in enumerate(batch_base):
                for mod_idx, mod_func in enumerate(batch_modified):
                    pairs.append((base_func, mod_func))
                    pair_indices.append((base_idx + i, mod_idx + j))
            
            inputs = tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to("cuda")
            
            with torch.no_grad(), autocast():
                outputs = model(**inputs)
                batch_probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            
            for (base_idx, mod_idx), score in zip(pair_indices, batch_probs):
                similarity_matrix[base_idx, mod_idx] = score
    
    return similarity_matrix

@measure_time
def analyze_results(similarity_matrix, base_functions, base_functions_types, modified_functions_types):
    accuracy_count = 0
    accuracy_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    
    ndcg_scores = []
    
    ndcg_metric = evaluate.load("ndcg")
    
    for i, (base_func, base_type) in enumerate(zip(base_functions, base_functions_types)):
        if i >= similarity_matrix.shape[0]:
            continue
            
        mod_scores = similarity_matrix[i]
        
        top_match_idx = np.argmax(mod_scores)
        predicted_type = modified_functions_types[top_match_idx]
        
        is_correct = (predicted_type == base_type)
        if is_correct:
            accuracy_count += 1
        
        accuracy_by_type[base_type]["total"] += 1
        if is_correct:
            accuracy_by_type[base_type]["correct"] += 1
        
        relevance_scores = [1 if t == base_type else 0 for t in modified_functions_types]
        
        try:
            ndcg_score = ndcg_metric.compute(
                predictions=mod_scores.tolist(),
                references=[relevance_scores],
                k=min(10, len(mod_scores))
            )['ndcg']
            ndcg_scores.append(ndcg_score)
        except Exception as e:
            print(f"Erreur lors du calcul du NDCG pour l'item {i}: {e}")
    
    global_accuracy = accuracy_count / len(base_functions) if base_functions else 0
    
    accuracy_by_type_values = {}
    for type_name, counts in accuracy_by_type.items():
        type_accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        accuracy_by_type_values[type_name] = type_accuracy
    
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
    
    return avg_ndcg, global_accuracy, accuracy_by_type, accuracy_by_type_values

def main():
    data, dataset_mesure = load_data()
    
    base_functions, base_functions_types, modified_functions, modified_functions_types = extract_functions(data)
    
    checkpoint_dirs = sorted(glob.glob('./results/checkpoint-25200'), key=lambda x: int(x.split('-')[-1]))
    
    # checkpoint_dirs = checkpoint_dirs[::3]
    
    ndcg_results = []
    accuracy_results = []
    accuracy_by_type_checkpoints = defaultdict(list)
    epochs = []
    
    print("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dirs[0])
    
    for checkpoint in tqdm(checkpoint_dirs, desc="Évaluation des checkpoints"):
        try:
            print(f"\nÉvaluation du checkpoint: {checkpoint}")
            
            epoch_num = int(checkpoint.split("-")[-1]) / 2520
            epochs.append(epoch_num)
            
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to("cuda")
            model.eval()
            
            similarity_matrix = compute_similarity_matrix(model, tokenizer, base_functions, modified_functions)

            ndcg_score, global_accuracy, accuracy_by_type, accuracy_by_type_values = analyze_results(
                similarity_matrix, base_functions, base_functions_types, modified_functions_types
            )

            print(f"  NDCG@10: {ndcg_score:.4f}")
            print(f"  Accuracy: {global_accuracy:.4f}")
            
            ndcg_results.append(ndcg_score)
            accuracy_results.append(global_accuracy)
            
            for type_name, acc in accuracy_by_type_values.items():
                accuracy_by_type_checkpoints[type_name].append(acc)
            
            print(f"  Global NDCG@10: {ndcg_score:.4f}")
            print(f"  Global Accuracy: {global_accuracy:.4f}")
            print("  Accuracy par type de fonction:")
            for ftype, acc in accuracy_by_type_values.items():
                print(f"    Type {ftype}: {acc:.4f} ({accuracy_by_type[ftype]['correct']}/{accuracy_by_type[ftype]['total']})")
            
        except Exception as e:
            print(f"Erreur sur {checkpoint} : {e}")
            continue
    
    if ndcg_results:
        plot_results(epochs, ndcg_results, accuracy_results, accuracy_by_type_checkpoints)

@measure_time
def plot_results(epochs, ndcg_results, accuracy_results, accuracy_by_type_checkpoints):
    if not epochs or not ndcg_results or not accuracy_results:
        print("Pas assez de données pour tracer les graphiques")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, ndcg_results, marker="o", linestyle='-', linewidth=2, markersize=6, label="NDCG@10")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("NDCG@10", fontsize=12)
    plt.title("Évolution du NDCG@10 par epoch", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("ndcg_curve.png", dpi=300)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracy_results, marker="o", linestyle='-', linewidth=2, markersize=6, color='green', label="Accuracy")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Évolution de l'Accuracy par epoch", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.show()

    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(accuracy_by_type_checkpoints)))
    
    for i, (type_name, acc_values) in enumerate(accuracy_by_type_checkpoints.items()):
        if acc_values:
            plt.plot(epochs, acc_values, marker="o", linestyle='-', 
                    linewidth=2, markersize=5, color=colors[i], 
                    label=f"Type {type_name}")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Évolution de l'Accuracy par type de fonction", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig("accuracy_by_type_curve.png", dpi=300)
    plt.show()
    
    final_accuracies = {t: values[-1] for t, values in accuracy_by_type_checkpoints.items()}
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(final_accuracies.keys(), final_accuracies.values(), color=colors[:len(final_accuracies)])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel("Type de fonction", fontsize=12)
    plt.ylabel("Accuracy finale", fontsize=12)
    plt.title("Accuracy finale par type de fonction", fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("final_accuracy_by_type.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()