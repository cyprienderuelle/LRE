import re
import torch
import json
import random
import concurrent.futures
import os
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== CONFIGURATION GLOBALE ====================

# Modèle et chemins
MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
BASE_PATH = "./"
DATASET_PATH = BASE_PATH + "List_functions.jsonl"
OUTPUT_PATH = BASE_PATH + "List_functions_positif.jsonl"

# Parallélisation
# NOTE: Le modèle deepseek-coder-6.7b-instruct est grand.
# L'utilisation de ProcessPoolExecutor est CORRECTE pour forcer le chargement
# du modèle sur différents GPU/VRAM, mais il faut s'assurer que l'environnement
# est propre. MAX_WORKERS = nombre de GPU.
MAX_WORKERS = 3 
DUPLICAT = 1 # Nombre d'échantillons positifs à générer par fonction ancre
MAX_FUNCTIONS_TO_PROCESS = 10 # Limite pour le test rapide

# ==================== FONCTIONS D'EXTRACTION ====================

def extract_functions_from_c_file(code: str):
    """
    Extraire les fonctions C en conservant signature + accolades. (Logique conservée)
    """
    functions = []
    length = len(code)
    i = 0
    if length > 10000:
        return functions
    while i < length:
        m = re.match(r'([a-zA-Z_][\w\s\*\(\),]*?)\s+([a-zA-Z_][\w]*)\s*\(', code[i:], re.S)
        if not m:
            i += 1
            continue
        ret_type, name = m.groups()
        start = i + m.start()
        j = i + m.end()
        paren_count = 1
        in_string = in_char = False
        escape = False
        while j < length and paren_count > 0:
            c = code[j]
            if c == '"' and not escape and not in_char:
                in_string = not in_string
            elif c == "'" and not escape and not in_string:
                in_char = not in_char
            elif not in_string and not in_char:
                if c == '(':
                    paren_count += 1
                elif c == ')':
                    paren_count -= 1
            escape = (c == '\\' and not escape)
            j += 1
        while j < length and code[j] in " \t\r\n":
            j += 1
        if j >= length or code[j] != '{':
            i = j
            continue
        brace_count = 1
        j += 1
        in_string = in_char = False
        escape = False
        while j < length and brace_count > 0:
            c = code[j]
            if c == '"' and not escape and not in_char:
                in_string = not in_string
            elif c == "'" and not escape and not in_string:
                in_char = not in_char
            elif not in_string and not in_char:
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
            escape = (c == '\\' and not escape)
            j += 1
        full_text = code[start:j]
        functions.append({
            'name': name,
            'return_type': ret_type.strip(),
            'full_text': full_text
        })
        i = j
    return functions

# ==================== FONCTION DE TRAVAIL (Worker) ====================

# Global pour stocker les modèles déjà chargés par le processus pour éviter la relecture
# REMARQUE : Dans ProcessPoolExecutor, chaque processus a sa propre copie de ces globales.
_MODEL_CACHE = {} 
_TOKENIZER_CACHE = {}

def get_model_and_tokenizer(model_id, gpu_device_index):
    """Charge le modèle/tokenizer une seule fois par processus/GPU."""
    
    # Clé de cache basée sur l'index du GPU
    cache_key = f"{model_id}_{gpu_device_index}"
    
    if cache_key in _MODEL_CACHE and cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key], _MODEL_CACHE[cache_key]

    # --- Initialisation du contexte GPU pour le processus ---
    if gpu_device_index is not None:
         # Affecter explicitement l'appareil pour ce processus
         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device_index)
         # target_device n'est plus utilisé pour le déplacement du modèle
         # target_device = torch.device(f"cuda:0") # Ancien code
    
    print(f"[Worker {os.getpid()}] Chargement du modèle sur CUDA_VISIBLE_DEVICES={gpu_device_index}")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto", 
            trust_remote_code=True,
            low_cpu_mem_usage=True 
        )
        
        # LIGNE CRÉANT LE CONFLIT: SUPPRIMÉE
        # model.to(target_device) # <-- C'est cette ligne qui causait l'erreur
        
        # Mise en cache
        _TOKENIZER_CACHE[cache_key] = tokenizer
        _MODEL_CACHE[cache_key] = model
        
        return tokenizer, model
        
    except Exception as e:
        print(f"[Worker {os.getpid()}] Erreur FATALE de chargement du modèle sur GPU {gpu_device_index}: {e}")
        if cache_key in _MODEL_CACHE: del _MODEL_CACHE[cache_key]
        if cache_key in _TOKENIZER_CACHE: del _TOKENIZER_CACHE[cache_key]
        raise

def worker_generate_positive_sample(func_data, model_id, duplicat, gpu_device_index=None):
    """
    Fonction de travailleur qui utilise le modèle chargé et génère les échantillons.
    """
    
    # 1. Chargement/Récupération du modèle
    try:
        tokenizer, model = get_model_and_tokenizer(model_id, gpu_device_index)
    except Exception:
        # L'erreur a déjà été loguée dans get_model_and_tokenizer
        return []

    # Le device est déterminé par le modèle après le chargement (device_map="auto")
    device = next(model.parameters()).device 
    
    # ------------------ Logique de génération ------------------
    def generate_single_sample(anchor_code, tokenizer_instance, model_instance, device_instance):
        
        techniques = [
            "Rename variables with different names",
            "Change loop structure (for/while)",
            "Reorder variable declarations",
            "Use different pointer/array notation",
            "Add intermediate variables"
        ]

        technique = random.choice(techniques)

        prompt = f"""Rewrite this C function keeping EXACTLY the same functionality.
Apply: {technique}

Original:
```c
{anchor_code}
```
Rewritten version:
"""
        # Utiliser `torch.no_grad()` pour l'inférence
        with torch.no_grad():
            inputs = tokenizer_instance(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device_instance) for k, v in inputs.items()}

            output_tokens = model_instance.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.12,
                pad_token_id=tokenizer_instance.pad_token_id,
                eos_token_id=tokenizer_instance.eos_token_id,
            )

        generated_text = tokenizer_instance.decode(output_tokens[0], skip_special_tokens=True)

        # Extraction et nettoyage du code généré (logique conservée)
        try:
            tmp = extract_functions_from_c_file(generated_text)
            if len(tmp) > 1:
                generated_text = tmp[1]["full_text"]
            elif len(tmp) == 1:
                generated_text = tmp[0]["full_text"]
            else:
                pass 
        except Exception:
            pass

        return generated_text
    # ------------------ Fin de la logique de génération ------------------

    results = []
    anchor = func_data['full_text']
    func_type = func_data['type']
    func_name = func_data.get('name', 'Inconnue')
    
    for _ in range(duplicat):
        try:
            positive = generate_single_sample(anchor, tokenizer, model, device)

            # Vérification basique de validité
            if 'error' in positive.lower() or not positive.strip().endswith('}'):
                # print(f"Réessayez ou utilisez l'ancre pour {func_name}")
                positive = anchor 

            results.append({
                'type': func_type,
                'positive': positive,
                'anchor': anchor
            })

        except Exception as e:
            # Gestion des erreurs de génération spécifiques au processus
            print(f"[Worker {func_name}] Erreur de génération : {e}")
            continue

    return results

# ==================== CHARGEMENT DES DONNÉES (Main) ====================

def load_functions(dataset_path):
    # Logique de chargement conservée
    fonctions = []
    count = 0
    print("Lecture des fonctions depuis le fichier JSONL...")
    try:
        with open(dataset_path, "r") as f:
            for line in f:
                func = json.loads(line)
                fonctions.append({
                    'name': func['name'],
                    'type': count, 
                    'full_text': func['full_text']
                })
                count += 1
    except FileNotFoundError:
        print(f"Erreur: Le fichier {dataset_path} n'a pas été trouvé.")
        return []

    print(f"{len(fonctions)} fonctions lues.")
    return fonctions

# ==================== EXÉCUTION PRINCIPALE ====================

if __name__ == '__main__':
    
    # Fixer les problèmes potentiels de fork/multiprocessing
    # On utilise "spawn" pour s'assurer que chaque processus enfant a un environnement propre
    torch.multiprocessing.set_start_method('spawn', force=True) 
    
    print(f"Démarrage du processus de génération en parallèle (Max {MAX_WORKERS} workers/GPU)...")
    
    # 1. Chargement des données
    fonctions = load_functions(DATASET_PATH)

    if not fonctions:
        print("Aucune fonction à traiter. Fin du script.")
        exit()
        
    if MAX_FUNCTIONS_TO_PROCESS > 0 and len(fonctions) > MAX_FUNCTIONS_TO_PROCESS:
        fonctions_to_process = fonctions[:MAX_FUNCTIONS_TO_PROCESS]
        print(f"Limite appliquée: Traitement des {MAX_FUNCTIONS_TO_PROCESS} premières fonctions.")
    else:
        fonctions_to_process = fonctions


    all_results = []
    
    # 2. Exécution en parallèle
    gpu_indices = [0, 1, 2] # Les index physiques de vos cartes (GTX 1080, 1080 Ti, P6000)
    
    # Utilisation d'un ProcessPoolExecutor pour le parallélisme CPU/GPU
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        futures = []
        for i, func in enumerate(fonctions_to_process):
            
            # Assigner un GPU cycliquement
            gpu_idx = gpu_indices[i % MAX_WORKERS]
            
            future = executor.submit(
                worker_generate_positive_sample, 
                func, 
                MODEL_ID, 
                DUPLICAT,
                gpu_idx # L'index GPU est passé comme argument
            )
            futures.append(future)
            
        print(f"{len(futures)} tâches de génération soumises à {MAX_WORKERS} workers...")

        # 3. Récupération des résultats
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                results = future.result()
                all_results.extend(results)
                
                print(f"-> Tâche {i+1}/{len(futures)} terminée. Résultats générés: {len(results)}. Total cumulé: {len(all_results)}")
                
            except Exception as e:
                print(f"Une tâche (Worker) a échoué pendant l'exécution: {e}")

    # 4. Sauvegarde finale
    print("\nÉcriture des résultats dans le fichier de sortie...")
    with open(OUTPUT_PATH, 'w') as f:
        for item in all_results:
            f.write(json.dumps(item) + '\n')

    print(f"\n✅ Génération terminée et sauvée dans {OUTPUT_PATH}")
    print(f"Total des échantillons (Anchor/Positive) : {len(all_results)}.")