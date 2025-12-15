import re
import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================= EXTRACTION DES FONCTIONS =======================
def extract_functions_from_c_file(code: str):
    """Extraire les fonctions C (signature + corps), en ignorant les chaînes et caractères"""
    functions = []
    length = len(code)
    i = 0

    if length > 10000:
        print("Fichier trop grand, saut de l'extraction.")
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
        in_string = in_char = escape = False

        # Fin signature
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

        # Compter accolades
        brace_count = 1
        j += 1
        in_string = in_char = escape = False

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
        functions.append({'name': name, 'return_type': ret_type.strip(), 'full_text': full_text})
        i = j

    return functions

# ======================= PATHS =======================
base_path = "./"
dataset_path = base_path + "List_functions.jsonl"
output_path = base_path + "List_functions_positif.jsonl"

# ======================= CHARGEMENT =======================
fonctions = []
count = 0
with open(dataset_path, "r") as f:
    for line in f:
        func = json.loads(line)
        fonctions.append({'name': func['name'], 'type': count, 'full_text': func['full_text']})
        count += 1

print(f"Total fonctions chargées: {len(fonctions)}\n")

# ======================= CHARGEMENT DU MODELE =======================
print("Chargement du modèle...")

model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map={
        0: "cuda:0",  # GTX 1080
        1: "cuda:1",  # GTX 1080 Ti
        2: "cuda:2"   # Quadro P6000
    },
    trust_remote_code=True,
)
model = torch.compile(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Modèle chargé sur multi-GPU.\n")

# ======================= GÉNÉRATION =======================
def generate_positive_sample(anchor_code):
    """Génère une version refactorisée (positive) de la fonction"""
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
Refactored version (same behavior, different code):
```c """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tmp = extract_functions_from_c_file(generated_text)
    if tmp:
        return tmp[0]["full_text"]
    else:
        print("⚠ Aucune fonction extraite, retour à l'ancre.")
        return anchor_code

# ======================= MULTI-THREAD & SAUVEGARDE =======================
from functools import partial

def worker(func):
    anchor = func['full_text']
    func_type = func['type']
    try:
        positive = generate_positive_sample(anchor)
        return {'type': func_type, 'anchor': anchor, 'positive': positive}
    except Exception as e:
        print(f"Erreur génération fonction {func['name']}: {e}")
        return None

duplicat = 1
max_count = 50  # nombre de fonctions à générer
results = []

print("Génération des positifs...\n")

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(worker, func) for func in fonctions[:max_count]]
    for fut in as_completed(futures):
        res = fut.result()
        if res:
            results.append(res)
            print(f"✅ Fonction générée: type {res['type']}")
        else:
            print("❌ Échec génération d'une fonction.")

# Sauvegarde finale
with open(output_path, 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()

print("\nTerminé ! Toutes les fonctions générées sont sauvegardées dans", output_path)
