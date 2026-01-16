import re
import json

def is_empty_function(code):
    # Cherche le dernier bloc entre accolades
    # Regex : une accolade ouvrante, suivie de n'importe quel espace/vide, 
    # suivie d'une accolade fermante à la fin de la chaîne.
    match = re.search(r'\{\s*\}', code)
    
    # On vérifie aussi si la fonction est très courte (ex: moins de 30 caractères)
    # ou si elle ne contient aucun mot-clé logique (if, return, etc.)
    if match and len(code.strip()) < 10000:
        # On vérifie qu'il n'y a pas de logique cachée
        logic_keywords = ['if', 'while', 'for', 'switch', 'return', '=']
        if not any(kw in code for kw in logic_keywords):
            return True
    return False

# Exemple d'utilisation dans ton tunnel de nettoyage
def clean_dataset(input_file, output_file):
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            if not is_empty_function(data.get('full_text', '')):
                f_out.write(json.dumps(data) + '\n')
            else:
                count += 1
    print(f"Supprimé {count} fonctions vides.")

clean_dataset('List_functions.jsonl', 'Cleaned_List_functions.jsonl')