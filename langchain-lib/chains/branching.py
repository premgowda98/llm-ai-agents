from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch


load_dotenv()

google_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

positive_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage("You are a helpful assistant"),
    ("human", "Give a thank you note for this positive feedback: {feedback}")
])

neutral_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage("You are a helpful assistant"),
    ("human", "Give a request for more details on this neutral feedback: {feedback}")
])


negative_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage("You are a helpful assistant"),
    ("human", "Give me the response address this negative feedback: {feedback}")
])

default_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage("You are a helpful assistant"),
    ("human", "Give a small note on this feedback: {feedback}")
])

classification_template = ChatPromptTemplate.from_messages([
    SystemMessage("You are a helpful assistant"),
    ("human", "Classify the sentiment of this feedback as positive, negative, neutral. feedback: {feedback}")
])

branches = RunnableBranch(
    (
        lambda x: 'positive' in x,
        positive_prompt_template | google_model | StrOutputParser()
    ),
    (
        lambda x: 'neutral' in x,
        neutral_prompt_template | google_model | StrOutputParser()
    ),
    (
        lambda x: 'negative' in x,
        negative_prompt_template | google_model | StrOutputParser()
    ),

    default_prompt_template | google_model | StrOutputParser()

)

classification_chain = classification_template | google_model | StrOutputParser()

chain = classification_chain | branches

p_feedback = "This macbook is excellent, i just love it. Will buy more of this"
n_feedback = "This is the worst apple product i ever brought, will never recommend it to others"
result = chain.invoke({"feedback": n_feedback})
print(result)