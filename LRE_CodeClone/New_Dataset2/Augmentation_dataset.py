import json
from Augmentation_c_fonctions import AugmentCode, samples

PATH_SAVE = "Cleaned_List_functions.jsonl"
RESULAT_FOLDER = "Resultat.jsonl"
NUMBER_OF_CLONE_POSITIF = 1

def create_dataset():

    with open(PATH_SAVE, "r") as f:
        functions = [json.loads(line)["full_text"] for line in f]

    print(f"Nombre de fonctions lues : {len(functions)}")
    count = 0
    count_fonction = 0

    with open(RESULAT_FOLDER, "a") as f:
        for func in functions:
            for i in range(NUMBER_OF_CLONE_POSITIF):
                try:
                    sample = samples(name='truc', function=func)
                    try:
                        query_function = AugmentCode(sample).function
                    except ValueError as e:
                        print(f"Erreur à l'itération {i} sur la fonction")
                        print(f"Nombre de lignes détectées : {len(func.splitlines())}")
                        print(f"Contenu problématique : {func[:100]}...") # Affiche le début
                        raise e # On arrête pour corriger
                        
                    
                    f.write(json.dumps({
                        "anchor": query_function,
                        "positif": func,
                        "type": count_fonction
                    }) + "\n")
                    count += 1
                except Exception as e:
                    print(f"Erreur à l'ifhetération {i}: {e}")

            count_fonction += 1

    print(str(count) + " triplet généré")
    print(str(count_fonction) + " fonctions traitées")

create_dataset()