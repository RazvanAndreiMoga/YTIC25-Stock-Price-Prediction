import openai


openai.api_base = "https://openrouter.ai/api/v1"
response = openai.Model.list()
print([m.id for m in response.data])


        