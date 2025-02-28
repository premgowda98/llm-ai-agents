from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import datetime
from wikipedia import summary
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

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
    )
]

prompt = hub.pull("hwchase17/structured-chat-agent")
# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_structured_chat_agent(
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