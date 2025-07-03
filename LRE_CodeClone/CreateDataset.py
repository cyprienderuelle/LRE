import json
import random
from CodeClone import *
from NewAugmentationNegatif import AugmentationNegatif
from NewAugmentationpositif import *
import re
import os

################################ VARIABLE ############################

DATASET_FOLDER = "samples_c/"
NUMBER_OF_CLONE_POSITIF = 25
NUMBER_OF_CLONE_NEGATIF = 25

######################################################################

def extract_functions(file_path):
    func_pattern = r'^\s*(\w[\w\s\*]*)\s+(\w+)\s*\(([^)]*)\)\s*{(.*?)^}\s*'

    with open(file_path, 'r') as f:
        content = f.read()

    functions = re.findall(func_pattern, content, re.MULTILINE | re.DOTALL)

    result = {}

    for return_type, func_name, params, func_body in functions:
      try:
        signature = f"{return_type} {func_name}({params})"
        code = f"{signature} {{\n{func_body.strip()}\n}}"

        result[func_name] = {}
        result[func_name]['code'] = code
      except:
        continue

    return result

"""
print("Extraction des fonctions terminée")
test = "#include \"array_max_min.h\"\
\
#include <stddef.h>\
\
void array_max_min(int tab[], size_t len, int *max, int *min)\
{\
    if (tab != NULL && len != 0)\
    {\
        *max = tab[0];\
        *min = tab[0];\
        for (size_t i = 0; i < len; i++)\
        {\
            if (tab[i] > *max)\
            {\
                *max = tab[i];\
            }\
            else if (tab[i] < *min)\
            {\
                *min = tab[i];\
            }\
        }\
    }\
}\
"

print(extract_functions("piscine/array_max_min/array_max_min.c")["array_max_min"])
"""

def get_c_files(folder):
    c_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".c"):
                c_files.append(os.path.join(root, file))
    return c_files

def create_dataset(folder):
    c_files = get_c_files(DATASET_FOLDER)

    # List of negative fonction
    negative_samples = []

    # List of training functions
    training_functions = []

    count = 0
    count_fonction = 0

    with open("dataset_splade_testt.jsonl", "w") as f:
        for filepath in c_files:
            functions = extract_functions(filepath)
            for function_name, jso in functions.items():
                try:
                    function = jso['code']
                    if len(function) < 150:
                        continue
                    for i in range(NUMBER_OF_CLONE_POSITIF):
                        try:
                            sample = samples(name=function_name, function=function)
                            query_function = AugmentCode(sample).function
                            f.write(json.dumps({
                                "query": query_function,
                                "document": function,
                                "label": 1,
                                "type": count_fonction
                            }) + "\n")
                            count += 1
                        except Exception as e:
                            print(f"Erreur à l'ifhetération {i}: {e}")


                    for i in range(NUMBER_OF_CLONE_NEGATIF):
                        if len(negative_samples) <= 10:
                            break
                        try:
                            negative_sample_tmp = negative_samples[random.randint(0, len(negative_samples) - 1)]
                            negative_sample = samples(name=negative_sample_tmp.name, function=negative_sample_tmp.function)
                            if (random.random() < 0.5):
                                query_function = AugmentationNegatif(negative_sample).function
                            else:
                                query_function = negative_sample.function

                            f.write(json.dumps({
                                "query": query_function,
                                "document": function,
                                "label": 0,
                                "type": count_fonction
                            }) + "\n")
                            count += 1
                        except Exception as e:
                            print(f"Erreur à l'itération {i}: {e}")
                
                    negative_samples.append(samples(name=function_name, function=function))

                except:
                    continue
                count_fonction += 1
        print(str(count) + " triplet généré")

if __name__ == "__main__":
    create_dataset(DATASET_FOLDER)
    print("Dataset created successfully.")