import json
import random
from CodeClone import *
import re
import os

################################ VARIABLE ############################

DATASET_FOLDER = "piscine/"
NUMBER_OF_CLONE_POSITIF = 5
NUMBER_OF_CLONE_NEGATIF = 10
KEY_WORD = ["main", "test"]

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

    t1 = transform1(window_size=10)
    t2 = transform2()
    t3 = tranform3()
    t4 = transform4(prefix="my_")
    t5 = transform5()
    lt = [t5, t2, t3, t4]
    lt2 = [t1]
    count_fonction = 0

    with open("otherdataset.jsonl", "w") as f:
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
                            query_function = lt[i % len(lt)](sample).function
                            f.write(json.dumps({
                                "query": query_function,
                                "document": function,
                                "label": 1,
                                "type": count_fonction
                            }) + "\n")
                            count += 1
                        except Exception as e:
                            print(f"Erreur à l'itération {i}: {e}")


                    for i in range(NUMBER_OF_CLONE_NEGATIF):
                        if len(negative_samples) <= 10:
                            break
                        try:
                            if False:
                                negative_sample_tmp = negative_samples[random.randint(0, len(negative_samples) - 1)]
                                negative_sample = samples(name=negative_sample_tmp.name, function=negative_sample_tmp.function)
                                query_function = lt[i % len(lt)](negative_sample).function
                            else:
                                negative_sample = samples(name=function_name, function=function)
                                query_function = lt2[i % len(lt2)](negative_sample).function

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

create_dataset(DATASET_FOLDER)