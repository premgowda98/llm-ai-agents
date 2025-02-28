from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

openai_model = ChatOpenAI(model="gpt-4o-mini")
google_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


messages = [
    SystemMessage("Solve the math problem"), # context messages
    HumanMessage("What is 78*78 ?")
]

user_input = int(input("Choose the model\n1.Openai\n2.GoogleAI\nEnter 1 or 2 : "))

if user_input==1:
    result = openai_model.invoke(messages)
elif user_input==2:
    result = google_model.invoke(messages)
else:
    # default to google
    result = google_model.invoke(messages)


print('Result:\n', result)
print('\nOutput:', result.content)

# Multiple AI Models