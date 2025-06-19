from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class HFLLMModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_id = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def predict_direction(self, price, volume, sentiment, trend):
        prompt = (
            f"The stock had an average price of {price:.2f}, volume of {volume:.2f}, "
            f"sentiment score of {sentiment:.2f}, and trend score of {trend:.2f}. "
            f"Will the price go up or down next?"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=10)
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

        if "up" in prediction:
            return 1
        elif "down" in prediction:
            return 0
        else:
            return 1 if sentiment > 0 else 0  # fallback
