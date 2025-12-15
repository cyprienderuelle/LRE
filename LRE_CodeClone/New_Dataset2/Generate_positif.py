import re
import torch
import json
import random
import concurrent.futures
import os
import threading
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 


# ==================== CONFIGURATION GLOBALE ====================

# MODÈLE POUR LA VITESSE : CodeLlama-3b
MODEL_ID = "codellama/CodeLlama-3b-Instruct-hf" 
BASE_PATH = "./"
DATASET_PATH = BASE_PATH + "List_functions.jsonl"
OUTPUT_PATH = BASE_PATH + "List_functions_positif.jsonl"

# Parallélisation
MAX_WORKERS = 3 
DUPLICAT = 1 
MAX_FUNCTIONS_TO_PROCESS = 0 # Mis à 0 pour traiter tout le jeu de données

# ==================== UTILITAIRE DE SAUVEGARDE INCÉMENTIELLE ====================

FILE_LOCK = threading.Lock() 

def save_incremental(results_list):
    """Sauvegarde les résultats d'une seule tâche dans le fichier de sortie."""
    if not results_list:
        return
        
    with FILE_LOCK:
        try:
            with open(OUTPUT_PATH, 'a') as f: 
                for item in results_list:
                    if isinstance(item, dict):
                        f.write(json.dumps(item) + '\n')
        except Exception as e:
            print(f"Erreur d'écriture incrémentielle : {e}")

# ==================== FONCTIONS D'EXTRACTION (Inchangé) ====================

def extract_functions_from_c_file(code: str):
    """
    Extraire les fonctions C en conservant signature + accolades.
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

_MODEL_CACHE = {} 
_TOKENIZER_CACHE = {}

def get_model_and_tokenizer(model_id, gpu_device_index):
    """
    Charge le modèle une seule fois par processus/GPU en utilisant le PID pour le cache.
    Ceci résout le problème 'CUDA out of memory'.
    """
    
    # Utiliser le PID comme clé de cache pour le processus courant
    cache_key = f"{model_id}_{os.getpid()}" 
    
    if cache_key in _MODEL_CACHE and cache_key in _TOKENIZER_CACHE:
        # Modèle déjà chargé dans ce processus, on le retourne.
        return _TOKENIZER_CACHE[cache_key], _MODEL_CACHE[cache_key]

    # Seulement si le modèle N'EST PAS dans le cache, on configure l'environnement et on charge.
    if gpu_device_index is not None:
         # Assurez-vous que le worker ne voit que son GPU dédié
         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device_index)
    
    # Déterminer le dtype de calcul optimal
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    print(f"[Worker {os.getpid()}] Chargement du modèle sur le GPU visible 0 (physique {gpu_device_index}) en 4-bit (compute_dtype={compute_dtype})...")

    try:
        # Configuration de la Quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype 
        )
        
        # Le tokenizer CodeLlama nécessite trust_remote_code=False
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False) 
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto", 
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
        
        # Ajout de Torch Compile pour la Vitesse
        try:
            model.eval() 
            model = torch.compile(model, mode="max-autotune")
            print(f"[Worker {os.getpid()}] Modèle compilé avec torch.compile (max-autotune).")
        except Exception as compile_error:
            print(f"[Worker {os.getpid()}] Avertissement: torch.compile a échoué: {compile_error}")

        
        _TOKENIZER_CACHE[cache_key] = tokenizer
        _MODEL_CACHE[cache_key] = model
        
        return tokenizer, model
        
    except Exception as e:
        print(f"[Worker {os.getpid()}] Erreur FATALE de chargement du modèle sur GPU {gpu_device_index}: {e}")
        # Nettoyage en cas d'échec
        if cache_key in _MODEL_CACHE: del _MODEL_CACHE[cache_key]
        if cache_key in _TOKENIZER_CACHE: del _TOKENIZER_CACHE[cache_key]
        raise

def worker_generate_positive_sample(func_data, model_id, duplicat, gpu_device_index=None):
    """
    Fonction de travailleur qui utilise le modèle chargé et génère les échantillons.
    """
    
    # 1. Chargement/Récupération du modèle (via le cache de PID)
    try:
        tokenizer, model = get_model_and_tokenizer(model_id, gpu_device_index)
    except Exception:
        return []

    device = next(model.parameters()).device 
    
    def generate_single_sample(anchor_code, tokenizer_instance, model_instance, device_instance):
        
        techniques = [
            "Rename variables with different names",
            "Change loop structure (for/while)",
            "Reorder variable declarations",
            "Use different pointer/array notation",
            "Add intermediate variables"
        ]

        technique = random.choice(techniques)

        # ------------------ ESPACE DU PROMPT ------------------
        # VEUILLEZ REMPLACER LA LIGNE CI-DESSOUS PAR VOTRE PROMPT COMPLET
        # Utilisez {anchor_code} pour insérer la fonction C et {technique} pour le refactoring.
        prompt = f"""Rewrite this C function keeping EXACTLY the same functionality.
                    Apply: {technique}

                    Original:
                    ```c
                    {anchor_code}
                    ```
                    Rewritten version: """ 
        
        # Exemple de ce que vous pourriez insérer :
        # prompt = f"""<|begin of instruction|>Rewrite the following C function: {anchor_code}. Apply {technique}. Give only the C code in the response.<|end of instruction|>"""
        
        # ------------------ FIN ESPACE DU PROMPT ------------------


        with torch.no_grad():
            inputs = tokenizer_instance(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device_instance) for k, v in inputs.items()}

            output_tokens = model_instance.generate(
                **inputs,
                max_new_tokens=200, 
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.12,
                pad_token_id=tokenizer_instance.pad_token_id,
                eos_token_id=tokenizer_instance.eos_token_id,
            )

        generated_text = tokenizer_instance.decode(output_tokens[0], skip_special_tokens=True)

        # Extraction et nettoyage du code généré
        try:
            # 1. Retirer le prompt de l'output (méthode générale)
            if generated_text.startswith(prompt.strip()):
                generated_text = generated_text[len(prompt.strip()):].strip()
                
            # 2. Extraire le code entre les balises ```c (très commun pour les modèles de code)
            code_match = re.search(r"```c\s*(.*?)\s*```", generated_text, re.DOTALL)
            if code_match:
                generated_text = code_match.group(1).strip()
            
            # 3. Si pas de balises, utiliser l'extracteur de fonction C (Logique robuste)
            else:
                 tmp = extract_functions_from_c_file(generated_text)
                 if len(tmp) >= 1:
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

            # Vérification basique de validité: si le résultat est vide ou semble invalide
            if not positive.strip() or 'error' in positive.lower() or not positive.strip().endswith('}'):
                positive = anchor 

            results.append({
                'type': func_type,
                'positive': positive,
                'anchor': anchor
            })

        except Exception as e:
            print(f"[Worker {func_name}] Erreur de génération : {e}")
            continue

    return results


# ==================== CHARGEMENT DES DONNÉES (Main) ====================

def load_functions(dataset_path):
    fonctions = []
    count = 0
    print("Lecture des fonctions depuis le fichier JSONL...")
    try:
        with open(dataset_path, "r") as f:
            for line in f:
                func = json.loads(line)
                fonctions.append({
                    'name': func.get('name', f'Func_{count}'),
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
    
    torch.multiprocessing.set_start_method('spawn', force=True) 
    
    print(f"Démarrage du processus de génération en parallèle (Max {MAX_WORKERS} workers/GPU)...")
    
    fonctions = load_functions(DATASET_PATH)

    if not fonctions:
        print("Aucune fonction à traiter. Fin du script.")
        exit()
        
    if MAX_FUNCTIONS_TO_PROCESS > 0 and len(fonctions) > MAX_FUNCTIONS_TO_PROCESS:
        fonctions_to_process = fonctions[:MAX_FUNCTIONS_TO_PROCESS]
        print(f"Limite appliquée: Traitement des {MAX_FUNCTIONS_TO_PROCESS} premières fonctions.")
    else:
        fonctions_to_process = fonctions
        print(f"Traitement de {len(fonctions_to_process)} fonctions.")


    processed_count = 0 
    total_samples = len(fonctions_to_process) * DUPLICAT
    
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print(f"Ancien fichier de résultats ({OUTPUT_PATH}) supprimé. Redémarrage propre.")

    # Exécution en parallèle
    gpu_indices = [0, 1, 2] # Les index physiques de vos cartes
    
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
                gpu_idx 
            )
            futures.append(future)
            
        print(f"{len(futures)} tâches de génération soumises à {MAX_WORKERS} workers...")

        # Récupération des résultats et Sauvegarde Incrémentielle
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                results = future.result()
                
                # SAUVEGARDE INCÉMENTIELLE
                save_incremental(results)
                
                processed_count += DUPLICAT
                
                print(f"-> Tâche {i+1}/{len(futures)} terminée. Samples générés: {len(results)}. Total cumulé: {processed_count}/{total_samples}")
                
            except Exception as e:
                print(f"Une tâche (Worker) a échoué pendant l'exécution: {e}")

    print(f"\n✅ Génération terminée.")
    
    try:
        final_count = sum(1 for line in open(OUTPUT_PATH))
        print(f"Total des échantillons sauvés dans {OUTPUT_PATH} : {final_count}.")
    except Exception:
        pass