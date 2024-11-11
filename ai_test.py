from llama_index.legacy.llms.ollama import Ollama


llm = Ollama(model="mistral", request_timeout=600.0)


prompt = (
  "Create a REST controller class in Java for a Spring Boot 3.2 application. "
  "This class should handle GET and POST requests, and include security and "
  "configuration annotations."
)

response = llm.complete(prompt)
print(response)