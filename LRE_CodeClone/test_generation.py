import time
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_id = "Salesforce/codegen2-1B_P"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = """
Règles : Réponds uniquement avec une structure JSON conforme au schéma AnswerSchema. Ne jamais inclure autre chose que le JSON, sans balise, sans texte autour.
Objectif : Générer des fonctions similaires à celle fournie dans 'Fonction', en appliquant les transformations listées.
Instructions :
1. Applique une ou plusieurs transformations de la liste.
2. Ne garde que le code utile.
3. Ne réponds qu'avec un JSON commençant par '{' et terminant par '}'.

Fonction :
int size() {
    return count; }

Transformations à appliquer :
1. Remplacer les types de données par des types équivalents (ex : 'int' -> 'long').
2. Ajouter des commentaires explicatifs.
3. Renommer la fonction et les variables.
4. Réorganiser le code.
5. Ajouter des espaces pour la lisibilité.
6. Ajouter de la journalisation.
7. Ajouter des assertions.
8. Gérer les erreurs.

AnswerSchema = [(
    "function": "string")]

Répondre uniquement avec une structure JSON respectant strictement le schéma "AnswerSchema" et commençant par "{" puis terminant par "}". Ne jamais encadrer la structure JSON par une balise.
"""

start_time = time.time()
outputs = generator(
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_k=50,
    num_return_sequences=1
)
end_time = time.time()

print(outputs[0]['generated_text'])
print(f"\nTemps de génération : {end_time - start_time:.2f} secondes")
