import re
import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

# ======================= FONCTION EXTRACTION =======================
def extract_functions_from_c_file(code: str):
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
        in_string = in_char = escape = False

        while j < length and paren_count > 0:
            c = code[j]
            if c == '"' and not escape and not in_char:
                in_string = not in_string
            elif c == "'" and not escape and not in_string:
                in_char = not in_char
            elif not in_string and not in_char:
                if c == '(': paren_count += 1
                elif c == ')': paren_count -= 1
            escape = (c == '\\' and not escape)
            j += 1

        while j < length and code[j] in " \t\r\n": j += 1
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
                if c == '{': brace_count += 1
                elif c == '}': brace_count -= 1
            escape = (c == '\\' and not escape)
            j += 1

        functions.append({'name': name, 'return_type': ret_type.strip(), 'full_text': code[start:j]})
        i = j
    return functions

# ======================= PATHS =======================
base_path = "./"
dataset_path = base_path + "List_functions.jsonl"
output_path = base_path + "List_functions_positif.jsonl"

# ======================= CHARGEMENT DATASET =======================
fonctions = []
print("Chargement des fonctions...")
with open(dataset_path, "r") as f:
    for count, line in enumerate(f):
        func = json.loads(line)
        fonctions.append({'name': func['name'], 'type': count, 'full_text': func['full_text']})
print(f"Total fonctions chargées: {len(fonctions)}\n")

# ======================= CHARGEMENT MODELE =======================
print("Chargement du modèle...")
model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = torch.compile(model)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
print(f"Modèle chargé sur {torch.cuda.device_count()} GPUs.\n")

# ======================= GENERATION =======================
gpus = [i for i in range(torch.cuda.device_count())]
gpu_cycle = cycle(gpus)

def generate_positive_sample(anchor_code, gpu):

    if len(anchor_code) > 1250:
        return (anchor_code, False)
    
    device = f"cuda:{gpu}"
    techniques = [
        "Rename variables with different names",
        "Change loop structure (for/while)",
        "Reorder variable declarations",
        "Use different pointer/array notation",
        "Add intermediate variables"
    ]
    technique = random.choice(techniques)
    prompt = f"""You are an expert C programmer. Rewrite the following C function in C language, keeping the EXACTLY SAME FUNCTION SIGNATURE and SEMANTIC. Apply: {technique}.

<|begin_of_original_function|>
{anchor_code}
<|end_of_original_function|>

<|begin_of_rewritten_function|>"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(len(anchor_code.split())*1.2 + len(prompt.split())),
            do_sample=True,
            temperature=0.7,
            top_k=50,
            num_beams=1,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text on GPU")
    print(generated_text)
    funcs = extract_functions_from_c_file(generated_text)
    
    if funcs:
        try:
            return (funcs[1]["full_text"], True)
        except IndexError:
            return (funcs[0]["full_text"], False)
    return (anchor_code, False)

# ======================= MULTI-THREAD / MULTI-GPU =======================
batch_size = 32
duplicat = 1
flush_size = 32
buffer = []

print("Génération des positifs...\n")
with open(output_path, 'w') as f, ThreadPoolExecutor(max_workers=len(gpus)*4) as executor:
    futures = []
    for i in range(0, len(fonctions), batch_size):
        batch = fonctions[i:i+batch_size]
        for func in batch:
            for _ in range(duplicat):
                gpu = next(gpu_cycle)
                futures.append(executor.submit(generate_positive_sample, func['full_text'], gpu))
        for future, func in zip(as_completed(futures), batch*duplicat):
            try:
                positive, success = future.result()
                buffer.append(json.dumps({
                    'type': func['type'],
                    'anchor': func['full_text'],
                    'positive': positive,
                    'success': success
                }))
                if len(buffer) >= flush_size:
                    f.write("\n".join(buffer) + "\n")
                    buffer = []
            except Exception as e:
                print(f"Erreur génération fonction {func['name']}: {e}")
    if buffer:
        f.write("\n".join(buffer) + "\n")

print("\nTerminé !")
