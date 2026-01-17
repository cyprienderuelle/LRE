import json
from collections import Counter

def get_tokens(text):
    # Simple tokenisation pour comparer le contenu
    return set(text.lower().split())

def analyze_dataset(filepath):
    overlap_scores = []
    lengths = []
    
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            ancre_tokens = get_tokens(data['ancre'])
            neg_tokens = get_tokens(data['neg'])
            
            # Calcul de l'intersection (mots communs entre ancre et négatif)
            intersection = ancre_tokens.intersection(neg_tokens)
            overlap_scores.append(len(intersection))
            lengths.append(len(neg_tokens))
            
    avg_overlap = sum(overlap_scores) / len(overlap_scores)
    avg_len = sum(lengths) / len(lengths)
    return avg_overlap, avg_len

# Comparaison
v1_overlap, v1_len = analyze_dataset('Dataset_InfoNCE_HardNeg.jsonl')
v2_overlap, v2_len = analyze_dataset('Dataset_InfoNCE_HardNeg_V2.jsonl')

print(f"--- Analyse de la difficulté des Négatifs ---")
print(f"V1 (Aléatoire/Simple) : Mots communs Ancre-Neg = {v1_overlap:.2f}")
print(f"V2 (Modèle 81%)      : Mots communs Ancre-Neg = {v2_overlap:.2f}")
print(f"\nDifférence d'overlap : {((v2_overlap/v1_overlap)-1)*100:.1f}%")