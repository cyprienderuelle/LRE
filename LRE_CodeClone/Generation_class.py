import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CodeCompletion:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def complete_code(self, prompt_code, max_tokens=200):
        output = self.generator(
            prompt_code,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.7
        )
        print("=== Code Complété ===")
        print(output[0]['generated_text'])

if __name__ == "__main__":
    completion = CodeCompletion("deepseek-ai/deepseek-coder-1.3b-instruct")
    
    prompt = """void lazy_sort(char data[][MAX_NAME_LEN], int LEN)\n    """
    
    completion.complete_code(prompt)
