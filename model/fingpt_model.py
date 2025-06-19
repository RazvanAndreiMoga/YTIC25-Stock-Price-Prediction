from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class FinGPTModel:
    def __init__(self):
        print("Loading FinGPTForecaster (LoRA fine-tuned on Llama 2 7B)...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
        )
        self.model.eval()


    def generate(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
