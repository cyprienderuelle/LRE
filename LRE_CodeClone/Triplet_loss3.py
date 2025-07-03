import os
import json
import random
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import ndcg_score, average_precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.cuda.amp import autocast

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(42)
np.random.seed(42)

class VariantAnalyzer:

    def __init__(self, path):
        self.dataset = self.load_dataset(path)
        self.functions_by_type = self.group_by_type()
        print(f"📦 Dataset: {len(self.dataset)} entrées | Types: {len(self.functions_by_type)}")

    def load_dataset(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith(".json"):
                return json.load(f)
            return [json.loads(line) for line in f if line.strip()]

    def group_by_type(self):
        grouped = defaultdict(lambda: {'base_documents': set(), 'variants': []})
        for item in self.dataset:
            grouped[item['type']]['base_documents'].add(item['document'])
            grouped[item['type']]['variants'].append(item)
        for k in grouped:
            grouped[k]['base_documents'] = list(grouped[k]['base_documents'])
        return grouped

    def create_mixed_variants(self, target_type, base_doc, n=100):
        same, other = [], []
        for t, data in self.functions_by_type.items():
            for v in data['variants']:
                if v['query'] == base_doc or v['document'] == base_doc:
                    continue
                v_copy = v.copy()
                v_copy['is_same_type'] = (t == target_type)
                (same if v_copy['is_same_type'] else other).append(v_copy)
        s, o = min(len(same), max(50, n//2)), n - min(len(same), max(50, n//2))
        selected = random.sample(same, s) + random.sample(other, min(o, len(other)))
        remaining = [v for v in (same + other) if v not in selected]
        return selected + random.sample(remaining, max(0, n - len(selected))) if remaining else selected

    def calculate_metrics(self, scores, labels):
        scores, labels = np.array(scores), np.array(labels)
        if not len(scores): return {}
        metrics = {
            'MAP': average_precision_score(labels, scores),
            'MRR': next((1/(i+1) for i, l in enumerate(labels[np.argsort(scores)[::-1]]) if l), 0)
        }
        for k in [5, 10, 20]:
            if len(scores) >= k:
                idx = np.argsort(scores)[::-1][:k]
                top_k = labels[idx]
                metrics[f'NDCG@{k}'] = ndcg_score([labels], [scores], k=k)
                metrics[f'Precision@{k}'] = np.mean(top_k)
                metrics[f'Recall@{k}'] = np.sum(top_k) / np.sum(labels) if np.sum(labels) else 0
        return metrics

    def analyze_with_model(self, model_type="custom", model_path="./results/checkpoint-25200"):
        print(f"\n🚀 Analyse avec modèle {model_type.upper()}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = (AutoModelForSequenceClassification if model_type == "custom" else AutoModel).from_pretrained(model_path).to(device).eval()
        encode_fn = self.encode_custom if model_type == "custom" else self.encode_splade

        all_metrics, by_type = [], {}
        for t, data in self.functions_by_type.items():
            print(f"\n🔍 Type: {t}")
            by_type[t] = []
            for base_doc in data['base_documents'][:3]:
                variants = self.create_mixed_variants(t, base_doc)
                if not variants: continue
                base_emb = encode_fn(base_doc, tokenizer, model, device)
                scores, labels = [], []
                for v in variants:
                    query_emb = encode_fn(v['query'], tokenizer, model, device)
                    score = self.compute_score(base_emb, query_emb, model_type)
                    scores.append(score)
                    labels.append(v['label'])
                metrics = self.calculate_metrics(scores, labels)
                metrics.update({
                    'same_type_variants': sum(v.get('is_same_type', False) for v in variants),
                    'total_variants': len(variants)
                })
                all_metrics.append(metrics)
                by_type[t].append(metrics)
        return all_metrics, by_type

    def encode_custom(self, text, tokenizer, model, device):
        with torch.no_grad(), autocast():
            inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True).to(device)
            return model(**inputs).logits

    def encode_splade(self, text, tokenizer, model, device):
        with torch.no_grad():
            inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True).to(device)
            h = model(**inputs).last_hidden_state
            return torch.log1p(torch.relu((h * inputs['attention_mask'].unsqueeze(-1)).max(1).values))

    def compute_score(self, a, b, model_type):
        return F.cosine_similarity(a, b).item() if model_type == "splade" else torch.softmax(b, dim=1)[0, 1].item()

    def compare_results(self, custom, splade, custom_by_type, splade_by_type):
        print("\n📊 COMPARAISON GLOBALE")
        def avg(metrics): return {k: np.mean([m[k] for m in metrics if k in m]) for k in metrics[0] if k not in ['same_type_variants', 'total_variants']}
        c_avg, s_avg = avg(custom), avg(splade)
        for metric in ['MAP', 'MRR', 'NDCG@5', 'Precision@5', 'Recall@5']:
            print(f"{metric:<12} | Custom: {c_avg.get(metric, 0):.4f} | SPLADE: {s_avg.get(metric, 0):.4f} | {'✅ Custom' if c_avg.get(metric, 0) > s_avg.get(metric, 0) else '✅ SPLADE'}")

        print("\n📌 COMPARAISON PAR TYPE")
        for t in set(custom_by_type) & set(splade_by_type):
            c_avg_t, s_avg_t = avg(custom_by_type[t]), avg(splade_by_type[t])
            print(f"Type {t}:")
            for m in ['MAP', 'NDCG@5']:
                print(f"  {m}: C={c_avg_t.get(m, 0):.4f} | S={s_avg_t.get(m, 0):.4f} {'📈' if c_avg_t.get(m, 0) > s_avg_t.get(m, 0) else '📉'}")

def main():
    path = "otherdataset.jsonl"
    analyzer = VariantAnalyzer(path)
    custom_metrics, custom_by_type = analyzer.analyze_with_model("custom", "./results/checkpoint-25200")
    splade_metrics, splade_by_type = analyzer.analyze_with_model("splade", "naver/splade_v2_max")
    analyzer.compare_results(custom_metrics, splade_metrics, custom_by_type, splade_by_type)

if __name__ == "__main__":
    main()
