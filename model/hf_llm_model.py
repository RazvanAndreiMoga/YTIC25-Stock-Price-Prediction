# model/hf_llm_model.py

import os
import requests
import json

class HFLLMModel:
    def __init__(self, model="openai/gpt-4o"):
        self.api_key = ""
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost",  # Optional: change for rankings
            "X-Title": "Stock Predictor App",    # Optional: change for rankings
            "Content-Type": "application/json"
        }

    def predict_direction(self, prompt):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial assistant. Only respond with 'UP' or 'DOWN'."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = requests.post(
                url=self.url,
                headers=self.headers,
                data=json.dumps(payload)
            )

            if response.status_code != 200:
                raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

            content = response.json()
            answer = content["choices"][0]["message"]["content"].strip().upper()
            return answer

        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {e}")
