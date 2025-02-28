from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()

curr_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curr_dir, "books", "india.txt")
persistent_dir = os.path.join(curr_dir, "db", "chroma_db")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if not os.path.exists(persistent_dir):
    loader = TextLoader(file_path)
    documents = loader.load()

    text_spliter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = text_spliter.split_documents(documents)

    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)

db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)
query = "Tell me about Siwalik Range"

retriver = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4}
)

relavant_docs = retriver.invoke(query)

# print(relavant_docs)


