from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

result = model.invoke("WHo are you")

print('Result:\n', result)
print('\nOutput:', result.content)