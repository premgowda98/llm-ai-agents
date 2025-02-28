from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel


load_dotenv()

google_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

messages = [
    SystemMessage("You are an expert product reviewer"),
    ("human", "Give me the main features of the product {product_name}")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

def analyse_pros(x):
    messages = [
        SystemMessage("You are an expert product reviewer"),
        ("human", "Given these features: {features} list the pros of the product")
    ]

    prompt_template = ChatPromptTemplate.from_messages(messages)

    return prompt_template.format_prompt(features=x)

def analyse_cons(x):
    messages = [
        SystemMessage("You are an expert product reviewer"),
        ("human", "Given these features: {features} list the cons of the product")
    ]

    prompt_template = ChatPromptTemplate.from_messages(messages)

    return prompt_template.format_prompt(features=x)


pros_branch_chain = RunnableLambda(lambda x: analyse_pros(x)) | google_model | StrOutputParser()
cons_branch_chain = RunnableLambda(lambda x: analyse_cons(x)) | google_model | StrOutputParser()

def combine_pros_cons(x,y): 
    return f"Pros: \n{x}\nCons: \n{y}\n"

"""
                                    --> Pros chain -> template -> model -> output ----
                                    |                                                |
prompt_template ----> Model ----->  |                                                | ---> Final Output
                                    |                                                |
                                    --> Cons chain -> template -> model -> output ----
"""

chain = (
    prompt_template
    | google_model
    | StrOutputParser()
    | RunnableParallel(branches={"pros":pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name":"MacBook m4 Pro"})
print(result)