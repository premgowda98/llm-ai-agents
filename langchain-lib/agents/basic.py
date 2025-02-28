from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import datetime

load_dotenv()

def get_current_time(*args, **kwargs):
    return datetime.datetime.now().strftime("%I:%M")

tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="To get current time"
    )
]

prompt = hub.pull("hwchase17/react")
llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

response = agent_executor.invoke({"input": "What is the time?"})

print(response)