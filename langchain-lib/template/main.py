from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage


load_dotenv()

google_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

template = "Tell me a joke on {joke}"
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"joke": "Virat Kholi"})
result = google_model.invoke(prompt)

print(result.content)

## With system prompt
messages = [
    SystemMessage("Provide short response, also include Sachin in this"),
    ("human", template)
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"joke": "Virat Kholi"})
result = google_model.invoke(prompt)

print(result.content)
