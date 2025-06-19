import openai

openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"
response = openai.Model.list()
print([m.id for m in response.data])

self.api_key = "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"
        