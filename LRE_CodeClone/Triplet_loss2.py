import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F
import json
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TripletDatasetHandler:
    
    def __init__(self, dataset_path):
        self.dataset = self.load_dataset(dataset_path)
        self.positive_pairs = [(item['query'], item['document']) for item in self.dataset if item['label'] == 1]
        self.negative_pairs = [(item['query'], item['document']) for item in self.dataset if item['label'] == 0]
        
        # Grouper par type pour créer des triplets plus cohérents
        self.functions_by_type = {}
        for item in self.dataset:
            func_type = item['type']
            if func_type not in self.functions_by_type:
                self.functions_by_type[func_type] = {'queries': [], 'documents': []}
            self.functions_by_type[func_type]['queries'].append(item['query'])
            self.functions_by_type[func_type]['documents'].append(item['document'])
    
    def load_dataset(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            if dataset_path.endswith('.jsonl'):
                return [json.loads(line.strip()) for line in f if line.strip()]
            else:
                try:
                    f.seek(0)
                    return json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    return [json.loads(line.strip()) for line in f if line.strip()]
    
    def create_triplets(self, num_triplets=100):
        triplets = []
        
        for _ in range(num_triplets):
            # Choisir une paire positive aléatoire
            if not self.positive_pairs:
                print("Aucune paire positive trouvée dans le dataset!")
                continue
                
            anchor, positive = random.choice(self.positive_pairs)
            
            negative_candidates = [doc for query, doc in self.negative_pairs]
            
            if not negative_candidates:
                all_docs = [item['document'] for item in self.dataset]
                negative_candidates = [doc for doc in all_docs if doc != positive and doc != anchor]
            
            if negative_candidates:
                negative = random.choice(negative_candidates)
                triplets.append((anchor, positive, negative))
        
        return triplets

def test_triplet_similarity_custom_model(triplets, model_path="./results/checkpoint-25200"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation de l'appareil: {device}")

    print(f"Chargement du modèle custom depuis {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    results = {
        'anchor_positive_scores': [],
        'anchor_negative_scores': [],
        'triplet_margins': [],
        'correct_rankings': 0,
        'total_triplets': len(triplets)
    }

    for i, (anchor, positive, negative) in enumerate(triplets):
        inputs_pos = tokenizer(
            [anchor], [positive],
            padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad(), autocast():
            outputs_pos = model(**inputs_pos)
            score_pos = torch.softmax(outputs_pos.logits, dim=1)[0, 1].item()

        inputs_neg = tokenizer(
            [anchor], [negative],
            padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad(), autocast():
            outputs_neg = model(**inputs_neg)
            score_neg = torch.softmax(outputs_neg.logits, dim=1)[0, 1].item()

        results['anchor_positive_scores'].append(score_pos)
        results['anchor_negative_scores'].append(score_neg)
        results['triplet_margins'].append(score_pos - score_neg)
        
        if score_pos > score_neg:
            results['correct_rankings'] += 1
            
        print(f"Triplet {i+1}: Pos={score_pos:.4f}, Neg={score_neg:.4f}, Margin={score_pos-score_neg:.4f}")

    return results

def test_triplet_similarity_splade(triplets, model_name="naver/splade_v2_max"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Chargement du modèle SPLADE {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def encode_with_splade(text):
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            max_pooled, _ = torch.max(masked_hidden, dim=1)
            sparse_embedding = torch.log1p(torch.relu(max_pooled))
        return sparse_embedding

    results = {
        'anchor_positive_scores': [],
        'anchor_negative_scores': [],
        'triplet_margins': [],
        'correct_rankings': 0,
        'total_triplets': len(triplets)
    }

    for i, (anchor, positive, negative) in enumerate(triplets):
        anchor_emb = encode_with_splade(anchor)
        positive_emb = encode_with_splade(positive)
        negative_emb = encode_with_splade(negative)
        
        score_pos = F.cosine_similarity(anchor_emb, positive_emb).item()
        score_neg = F.cosine_similarity(anchor_emb, negative_emb).item()

        results['anchor_positive_scores'].append(score_pos)
        results['anchor_negative_scores'].append(score_neg)
        results['triplet_margins'].append(score_pos - score_neg)
        
        if score_pos > score_neg:
            results['correct_rankings'] += 1
            
        if i % 10 == 0:
            print(f"Triplet {i+1}: Pos={score_pos:.4f}, Neg={score_neg:.4f}, Margin={score_pos-score_neg:.4f}")

    return results

def print_results(results, model_name):
    
    print(f"\n--- Résultats pour {model_name} ---")
    print(f"Nombre de triplets testés: {results['total_triplets']}")
    print(f"Rankings corrects: {results['correct_rankings']}/{results['total_triplets']} ({results['correct_rankings']/results['total_triplets']*100:.2f}%)")
    
    import numpy as np
    print(f"Score moyen ancre-positif: {np.mean(results['anchor_positive_scores']):.4f} ± {np.std(results['anchor_positive_scores']):.4f}")
    print(f"Score moyen ancre-négatif: {np.mean(results['anchor_negative_scores']):.4f} ± {np.std(results['anchor_negative_scores']):.4f}")
    print(f"Marge moyenne: {np.mean(results['triplet_margins']):.4f} ± {np.std(results['triplet_margins']):.4f}")

if __name__ == "__main__":

    dataset_path = "otherdataset.jsonl"
    
    try:
        triplet_handler = TripletDatasetHandler(dataset_path)
        print(f"Dataset chargé avec succès: {len(triplet_handler.dataset)} entrées")
        
        positive_count = len(triplet_handler.positive_pairs)
        negative_count = len(triplet_handler.negative_pairs)
        print(f"Paires positives: {positive_count}")
        print(f"Paires négatives: {negative_count}")
        
        if positive_count == 0:
            print("ERREUR: Aucune paire positive trouvée dans le dataset!")
            print("Vérifiez que votre dataset contient des entrées avec label=1")
            exit()
        
        num_triplets = min(10000, positive_count)
        triplets = triplet_handler.create_triplets(num_triplets)
        
        if not triplets:
            print("Aucun triplet n'a pu être créé. Vérifiez votre dataset.")
            exit()
        
        print(f"Créé {len(triplets)} triplets pour les tests.")
        
        if triplets:
            anchor, positive, negative = triplets[0]
            print(f"\nExemple de triplet:")
            print(f"Ancre: {anchor[:100]}...")
            print(f"Positif: {positive[:100]}...")
            print(f"Négatif: {negative[:100]}...")
        
        print("\n" + "="*50)
        results_custom = test_triplet_similarity_custom_model(triplets)
        print_results(results_custom, "Modèle Custom Fine-tuné")
        
        print("\n" + "="*50)
        results_splade = test_triplet_similarity_splade(triplets)
        print_results(results_splade, "SPLADE")
        
        print("\n" + "="*50)
        print("COMPARAISON DES MODÈLES")
        print(f"Accuracy Custom: {results_custom['correct_rankings']/results_custom['total_triplets']*100:.2f}%")
        print(f"Accuracy SPLADE: {results_splade['correct_rankings']/results_splade['total_triplets']*100:.2f}%")
    
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        print("Vérifiez le format de votre dataset et le chemin du fichier.")