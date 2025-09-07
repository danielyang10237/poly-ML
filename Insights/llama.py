from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1")

response = llm.invoke("List five countries in Africa")
print(response)