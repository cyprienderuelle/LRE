import re
import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_functions_from_c_file(code: str):
    """
    Extraire les fonctions C en conservant signature + accolades,
    en ignorant // ou /* */ à l'intérieur des chaînes.
    """
    functions = []
    length = len(code)
    i = 0

    if length > 10000:
        print("Fichier trop grand, saut de l'extraction.")
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

base_path = "./"
dataset_path = base_path + "List_functions.jsonl"
output_path = base_path + "List_functions_positif.jsonl"

fonctions = []
fonctions_by_type = {}
count = 0

with open(dataset_path, "r") as f:
    for line in f:
        func = json.loads(line)

        func_type = count
        count += 1

        fonctions.append({
            'name': func['name'],
            'type': func_type,
            'full_text': func['full_text']
        })

print("\nChargement du modèle...")
model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
# Alternative: "codellama/CodeLlama-7b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = torch.compile(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Modèle chargé\n")

# ======================= GÉNÉRATION DES TRIPLETS =======================
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
    
    batch_prompts = [f"Rewrite this C function keeping EXACTLY the same functionality.\nApply: {technique}\nOriginal:\n```c\n{code}\n```\nRefactored version (same behavior, different code):\n```c" for code in anchor_codes]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
        eos_token_id=tokenizer.eos_token_id,
    )


    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    tmp = extract_functions_from_c_file(generated_text)
    if len(tmp) > 0:
        generated_text = tmp[1]["full_text"]
    else:
      print(generated_text)

    return generated_text

# ======================= GÉNÉRATION ET SAUVEGARDE =======================
print("Génération des positifs...\n")

duplicat = 1
count2 = 0

with open(output_path, 'w') as f:
    for func in fonctions:
        if count2 == 10 and count2 > 0:
            break
        count2 += 1
        # Ancre
        anchor = func['full_text']
        func_type = func['type']

        for i in range(duplicat):
          try:
              
              # Positive
              positive = generate_positive_sample(anchor)

              # Vérification basique de validité
              if 'error' in positive.lower():
                  positive = anchor

              tmp = {
                  'type': func_type,
                  'positive': positive,
                  'anchor': anchor
              }

              f.write(json.dumps(tmp) + '\n')

          except Exception as e:
              continue