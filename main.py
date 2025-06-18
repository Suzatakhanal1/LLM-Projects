# # from langchain_ollama import OllamaLLM

# # model = OllamaLLM(model = "llama3")

# # result = model.chat(input = "Tell me top 5 places to visit in kathmandu")
# # print(result)


# import ollama
# model = ollama(model = "llama3")
# result = model.chat("Tell me top 5 places to visit in kathmandu")


# import ollama


# response = ollama.chat(model='llama3', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])

import ollama

# response = ollama.generate(model="llama3.2", prompt="What is an apple?")
# print(response['response'])

# model = 'gemma'
# messages = [
#     {"role": "system", "content": "You are a teacher."},
#     {"role": "user", "content": "What is AI?"},
# ]

# response = ollama.chat(model = model, messages = messages)
# print(response['message']['content'])

text = "Nepal, officially the Federal Democratic Republic of Nepal, is a landlocked country in South Asia. It is mainly situated in the Himalayas, but also includes parts of the Indo-Gangetic Plain. It borders the Tibet Autonomous Region of China to the north, and India to the south, east, and west, while it is narrowly separated from Bangladesh by the Siliguri Corridor, and from Bhutan by the Indian state of Sikkim."
# prompt = f"Summarize the following text in one sentence:{text}"
# prompt = f"Summarize the following text in one sentence:\n\"\"\"\n{text}\n\"\"\""
prompt = f"Summarize the following text in one sentence:\n{text}\n"
result = ollama.generate(model = 'llama3.2', prompt=prompt)
print(prompt)
print("Summary:", result['response'])

