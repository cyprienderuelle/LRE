import json
import matplotlib.pyplot as plt

# Chemin vers ton fichier .jsonl
file_path = "List_functions.jsonl"

total_chars = 0
total_lines = 0
total_functions = 0
lines_per_function = []
chars_per_function = []

# Lecture du fichier
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue  # ignorer les lignes vides
        item = json.loads(line)
        code = item.get("full_text", "")
        num_chars = len(code)
        num_lines = code.count("\n") + 1

        total_chars += num_chars
        total_lines += num_lines
        total_functions += 1
        lines_per_function.append(num_lines)
        chars_per_function.append(num_chars)

# Calculs
avg_chars = total_chars / total_functions if total_functions else 0
avg_lines = total_lines / total_functions if total_functions else 0

# Affichage des statistiques
print(f"Nombre de fonctions : {total_functions}")
print(f"Nombre total de lignes : {total_lines}")
print(f"Taille totale (caractères) : {total_chars}")
print(f"Taille moyenne par fonction (caractères) : {avg_chars:.2f}")
print(f"Nombre moyen de lignes par fonction : {avg_lines:.2f}")

# Histogramme du nombre de lignes
plt.figure(figsize=(10, 5))
plt.hist(lines_per_function, bins=range(0, max(lines_per_function)+5, 5),
         color='skyblue', edgecolor='black')
plt.title("Répartition du nombre de lignes par fonction")
plt.xlabel("Nombre de lignes")
plt.ylabel("Nombre de fonctions")
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histogramme du nombre de caractères
plt.figure(figsize=(10, 5))
plt.hist(chars_per_function, bins=20, color='lightgreen', edgecolor='black')
plt.title("Répartition du nombre de caractères par fonction")
plt.xlabel("Nombre de caractères")
plt.ylabel("Nombre de fonctions")
plt.grid(axis='y', alpha=0.75)
plt.show()
