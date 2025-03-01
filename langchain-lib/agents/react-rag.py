from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain import chains
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import datetime
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from wikipedia import summary
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
load_dotenv()

curr_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(curr_dir, "..","rag","db", "chroma_db")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

retriver = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4}
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

contextualized_system_prompt = (
    "Given a chat history and the latest user question which might reference context in chat history"
    "formulate a standalone question which can be understood without chat history"
    "DO NOT answer the question, just reformulate it if needed and otherwise return it as is"
)

context_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualized_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriver = chains.create_history_aware_retriever(
    llm, retriver, context_prompt
)

qa_system_prompt = (
    "You are an AI Assistant for question answering tasks"
    "Use the following retrieved context to answer the question."
    "If you don't know the answer, just say you don't know. Use 3 sentencemax and keep answer short"
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriver, question_answer_chain)

prompt = hub.pull("hwchase17/react")


def get_current_time(*args, **kwargs):
    return datetime.datetime.now().strftime("%I:%M")

def search_wiki(query):
    try:
        return summary(query, sentences=2)
    except:
        return "Not able to get the question"

tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Use when you need to get current time"
    ),
    Tool(
        name="Wikipedia",
        func=search_wiki,
        description="Use when you need any information"
    ),
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="Use when you need to answer questions about the context"
    )
]

# llm = ChatOpenAI(model="gpt-4o-mini")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory
)

initial_message = "You are AI Assistant that can provide helpful answers, using available tools"
memory.chat_memory.add_message(SystemMessage(initial_message))

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    memory.chat_memory.add_user_message(user_input)
    response = agent_executor.invoke({"input": user_input})
    memory.chat_memory.add_ai_message(response["output"])

    print(response["output"])