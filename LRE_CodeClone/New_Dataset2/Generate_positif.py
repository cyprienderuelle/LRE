import re
import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ======================= FONCTION EXTRACTION =======================
def extract_functions_from_c_file(code: str):
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
        functions.append({
            'name': name,
            'return_type': ret_type.strip(),
            'full_text': full_text
        })
        i = j
    return functions

# ======================= PATHS =======================
base_path = "./"
dataset_path = base_path + "List_functions.jsonl"
output_path = base_path + "List_functions_positif.jsonl"

# ======================= CHARGEMENT DATASET =======================
fonctions = []
count = 0
print("Chargement des fonctions...")
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
print(f"Total fonctions chargées: {len(fonctions)}\n")

# ======================= CHARGEMENT MODELE =======================
print("Chargement du modèle...")
#model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = torch.compile(model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Modèle chargé sur multi-GPU.\n")

# ======================= GENERATION =======================
def generate_positive_sample(anchor_code):
    techniques = [
        "Rename variables with different names",
        "Change loop structure (for/while)",
        "Reorder variable declarations",
        "Use different pointer/array notation",
        "Add intermediate variables"
    ]
    technique = random.choice(techniques)
    prompt = f"""You are an expert C programmer. Rewrite the following C function in C language, keeping the EXACTLY SAME FUNCTION SIGNATURE and EXACTLY SAME SEMANTIC BEHAVIOR. Apply the refactoring technique: {technique}.

                    <|begin_of_original_function|>
                    {anchor_code}
                    <|end_of_original_function|>

                    <|begin_of_rewritten_function|>"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.8,
            num_beams=1,
            do_sample=True,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== Generated Function ===")
    print(generated_text)
    tmp = extract_functions_from_c_file(generated_text)
    if tmp:
        return tmp[0]["full_text"]
    else:
        return anchor_code

# ======================= GENERATION PAR BATCH =======================
batch_size = 1
duplicat = 1
max_count = 1  # traiter toutes les fonctions
print("Génération des positifs...\n")

buffer = []
flush_size = 32  # écrire tous les 32 résultats

with open(output_path, 'w') as f:
    for i in tqdm(range(0, min(len(fonctions), max_count), batch_size)):
        batch = fonctions[i:i+batch_size]
        for func in batch:
            for _ in range(duplicat):
                try:
                    positive = generate_positive_sample(func['full_text'])
                    tmp = {
                        'type': func['type'],
                        'positive': positive,
                        'anchor': func['full_text']
                    }
                    buffer.append(json.dumps(tmp))
                    if len(buffer) >= flush_size:
                        f.write("\n".join(buffer) + "\n")
                        buffer = []
                except Exception as e:
                    print(f"Erreur génération fonction {func['name']}: {e}")
    # flush final
    if buffer:
        f.write("\n".join(buffer) + "\n")

print("\nTerminé !")
