import re
import torch
import json
import random
import concurrent.futures
import os # Importé pour potentiellement gérer l'affectation des GPU, bien que "device_map='auto'" soit souvent suffisant.

# ==================== CONFIGURATION GLOBALE ====================

# Modèle et chemins
MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
BASE_PATH = "./"
DATASET_PATH = BASE_PATH + "List_functions.jsonl"
OUTPUT_PATH = BASE_PATH + "List_functions_positif.jsonl"

# Parallélisation
# Ajustez MAX_WORKERS en fonction du nombre de GPU et de leur VRAM.
# Ici, 3 GPU sont disponibles.
MAX_WORKERS = 3 
DUPLICAT = 1 # Nombre d'échantillons positifs à générer par fonction ancre
MAX_FUNCTIONS_TO_PROCESS = 10 # Limite pour le test rapide (comme dans votre script original)

# ==================== FONCTIONS D'EXTRACTION (Mises à jour) ====================

def extract_functions_from_c_file(code: str):
    """
    Extraire les fonctions C en conservant signature + accolades,
    en ignorant // ou /* */ à l'intérieur des chaînes.
    """
    functions = []
    length = len(code)
    i = 0

    if length > 10000:
        # Laisser ce print ici peut aider au débogage du worker
        # print("Fichier trop grand, saut de l'extraction.")
        return functions

    while i < length:
        # Chercher une signature de fonction
        m = re.match(r'([a-zA-Z_][\w\s\*\(\),]*?)\s+([a-zA-Z_][\w]*)\s*\(', code[i:], re.S)
        if not m:
            i += 1
            continue

        ret_type, name = m.groups()
        start = i + m.start()
        j = i + m.end()
        paren_count = 1

        # Fin de la parenthèse de la signature
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

        # Skip whitespace
        while j < length and code[j] in " \t\r\n":
            j += 1

        if j >= length or code[j] != '{':
            i = j
            continue

        # Compter accolades hors chaînes
        brace_count = 1
        body_start = j
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

def worker_generate_positive_sample(func_data, model_id, duplicat, gpu_device_index=None):
    """
    Fonction de travailleur qui charge le modèle et génère les échantillons
    pour une seule fonction 'ancre'.
    """
    # Importation locale du tokenizer et du modèle pour le processus
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Si un index GPU est fourni, on le force pour ce processus.
    # Ceci est OPTIONNEL car 'device_map="auto"' est souvent plus flexible.
    if gpu_device_index is not None:
         # Affecter explicitement l'appareil pour ce processus
         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device_index)

    try:
        # Charger le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Charger le modèle (device_map="auto" le placera sur un GPU disponible)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
    except Exception as e:
        # Ce processus ne peut pas charger le modèle (souvent un problème de VRAM)
        print(f"[Worker] Erreur de chargement du modèle pour la fonction {func_data.get('name', 'Inconnue')} : {e}")
        return []

    # Récupérer l'appareil réel où le modèle a été placé
    device = model.device 
    
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
```c
"""
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

        # Extraction et nettoyage du code généré
        try:
            tmp = extract_functions_from_c_file(generated_text)
            
            # Votre code original utilisait tmp[1]. 
            # Je conserve cette logique, mais le plus souvent, c'est tmp[0] le premier résultat.
            if len(tmp) > 1:
                generated_text = tmp[1]["full_text"]
            elif len(tmp) == 1:
                generated_text = tmp[0]["full_text"]
            else:
                # Échec de l'extraction, on logue le texte brut
                # print(f"Échec extraction : {generated_text}") 
                pass # Conserver le texte brut si rien n'est extrait

        except Exception:
            # En cas d'erreur dans l'extraction
            # print("Erreur durant l'extraction après génération.")
            pass

        return generated_text
    # ------------------ Fin de la logique de génération ------------------

    results = []
    anchor = func_data['full_text']
    func_type = func_data['type']
    
    for _ in range(duplicat):
        try:
            positive = generate_single_sample(anchor, tokenizer, model, device)

            # Vérification basique de validité
            if 'error' in positive.lower() or not positive.strip().endswith('}'):
                positive = anchor # Revenir à l'ancre si la génération semble mauvaise

            results.append({
                'type': func_type,
                'positive': positive,
                'anchor': anchor
            })

        except Exception as e:
            # Gestion des erreurs de génération spécifiques au processus
            # print(f"[Worker] Erreur de génération pour {func_data['name']} : {e}")
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
                    'name': func['name'],
                    'type': count, # Utiliser le compteur comme type
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
    
    print(f"Démarrage du processus de génération en parallèle (Max {MAX_WORKERS} workers/GPU)...")
    
    # 1. Chargement des données
    fonctions = load_functions(DATASET_PATH)

    if not fonctions:
        print("Aucune fonction à traiter. Fin du script.")
        exit()
        
    # Appliquer la limite de fonctions si définie
    if MAX_FUNCTIONS_TO_PROCESS > 0 and len(fonctions) > MAX_FUNCTIONS_TO_PROCESS:
        fonctions_to_process = fonctions[:MAX_FUNCTIONS_TO_PROCESS]
        print(f"Limite appliquée: Traitement des {MAX_FUNCTIONS_TO_PROCESS} premières fonctions.")
    else:
        fonctions_to_process = fonctions


    all_results = []
    
    # 2. Exécution en parallèle
    
    # Utilisation d'un ProcessPoolExecutor pour le parallélisme CPU/GPU
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        futures = []
        # Vous pouvez assigner un index GPU spécifique si vous le souhaitez
        # Sinon, device_map="auto" se débrouillera
        gpu_indices = [0, 1, 2] # Correspond aux 3 GPU
        
        for i, func in enumerate(fonctions_to_process):
            # Assigner un GPU cycliquement (optimisation optionnelle)
            gpu_idx = gpu_indices[i % MAX_WORKERS] if MAX_WORKERS == 3 else None
            
            future = executor.submit(
                worker_generate_positive_sample, 
                func, 
                MODEL_ID, 
                DUPLICAT,
                gpu_idx
            )
            futures.append(future)
            
        print(f"{len(futures)} tâches de génération soumises à {MAX_WORKERS} workers...")

        # 3. Récupération des résultats au fur et à mesure de leur achèvement
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                # Les résultats sont des listes de dictionnaires
                results = future.result()
                all_results.extend(results)
                
                # Affichage de la progression
                print(f"-> Tâche {i+1}/{len(futures)} terminée. Résultats générés: {len(results)}. Total cumulé: {len(all_results)}")
                
            except Exception as e:
                print(f"Une tâche (Worker) a échoué: {e}")

    # 4. Sauvegarde finale
    print("\nÉcriture des résultats dans le fichier de sortie...")
    with open(OUTPUT_PATH, 'w') as f:
        for item in all_results:
            f.write(json.dumps(item) + '\n')

    print(f"\n✅ Génération terminée et sauvée dans {OUTPUT_PATH}")
    print(f"Total des échantillons (Anchor/Positive) : {len(all_results)}.")