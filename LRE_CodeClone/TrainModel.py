from datasets import Dataset
import json
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EVAL_DATASET_POURCENTAGE = 20/100

open_file = open('newdataset/allfunctions&variants.jsonl')
dataset = [json.loads(line) for line in open_file]

dataset_training = []
dataset_eval = []
dataset_mesure = []

for i in range(0, len(dataset), 10):
    block = dataset[i:i+10]
    new_block = []
    for i in range(1, len(block)):
        new_block.append({k: block[i][k] for k in ["query", "document", "label"]})

    dataset_eval.append(new_block[1])
    dataset_training.extend(new_block[2:8])
    dataset_mesure.append(block[0])

f = open('dataset_splade_mesure.jsonl', 'w+')
for line in dataset_mesure:
    f.write(json.dumps(line) + "\n")
f.close()

print("ok")

dataset_training = Dataset.from_list(dataset_training)
dataset_eval = Dataset.from_list(dataset_eval)

tokenizer = AutoTokenizer.from_pretrained("naver/splade_v2_max")

def tokenize_function(examples):
    return tokenizer(examples['query'], examples['document'], truncation=True, padding="max_length")

print("length dataset training : ", len(dataset_training))
print("length dataset eval : ", len(dataset_eval))
print("length dataset mesure : ", len(dataset_mesure))

tokenized_datasets_training = dataset_training.map(tokenize_function, batched=True).with_format("torch")
tokenized_datasets_eval = dataset_eval.map(tokenize_function, batched=True).with_format("torch")

model = AutoModelForSequenceClassification.from_pretrained("naver/splade_v2_max", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results_second',
    eval_steps=1000,
    logging_dir='./logs_second',
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    fp16=True,
    num_train_epochs=10,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets_training,
    eval_dataset=tokenized_datasets_eval,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("./trained_splade_model_second")
tokenizer.save_pretrained("./trained_splade_model_second")