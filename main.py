from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


model =OllamaLLM(model = "gemma3:1b-it-qat")

template ="""
You are a helpful expert in answering questions about the Python programming language. But within 3 sentences. One example if possible.
Here will be questions : {question}"""



prompt = ChatPromptTemplate.from_template("Questions: {question}")

chain =prompt| model

result = chain.invoke({"question": "What is a lambda function in Python?"})

print(result)
