from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence


load_dotenv()

google_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

messages = [
    SystemMessage("Provide short response, also include Sachin in this"),
    ("human", "Tell me a joke on {company}")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

upper_case = RunnableLambda(lambda x: x.upper())


# LangChain Expression Language
chain = prompt_template | google_model | StrOutputParser() | upper_case
result = chain.invoke({"company": "OpenAI"})
print("*"*50)
print(result)
print("*"*50)


"""
Breaking down chain working
"""

prompt_template = ChatPromptTemplate.from_messages(messages)

format_input = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: google_model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_input, middle=[invoke_model], last=parse_output)
result = chain.invoke({"company": "OpenAI"})
print("*"*50)
print(result)
print("*"*50)