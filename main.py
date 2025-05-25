from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


template = """
Answer the following question as best as you can. You are a helpful assistant.
Here is the conversation history: {context}
Question: {input}
Answer:

"""
model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    print("**********Welcome to the conversation! Type 'exit' or 'quit' to end the chat***********")
    context = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the conversation.")
            break
        context.append(user_input)
        response = chain.invoke({"context": context, "input": user_input})
        print(f"Assistant:", response)
        context.append(response)
        
if __name__ == "__main__":
    handle_conversation()
