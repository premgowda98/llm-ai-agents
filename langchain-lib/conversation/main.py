from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("*"*50)
user_input = int(input("Choose the model\n1.Openai\n2.GoogleAI\nEnter 1 or 2 : "))

if user_input==1:
    model = ChatOpenAI(model="gpt-4o-mini")
elif user_input==2:
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
else:
    # default to google
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

print("*"*50)

chat_history = []
system = SystemMessage("You are an AI Assistant")
chat_history.append(system)

while True:
    query = input("You: ")

    if query == "quit":
        print("\nThanks for chatting")
        print("*"*50)
        break

    chat_history.append(HumanMessage(query))
    result = model.invoke(chat_history)

    print('ChatBot: ', result.content)
    chat_history.append(AIMessage(result.content))

