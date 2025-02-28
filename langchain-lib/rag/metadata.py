from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

"""
Text Splitters

1. Character Splinting
2. Sentences Splinting
3. Token Splinting
4. Recursive Character Splinting
5. Custom Splinting 

Retrier

1. Similarity Search
2. Max Marginal Relevance
    Similarity with adjacent documents
2. Similarity Score Threshold
"""


load_dotenv()

curr_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curr_dir, "books")
persistent_dir = os.path.join(curr_dir, "db", "chroma_db_metadata")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if not os.path.exists(persistent_dir):
    books_file = [i for i in os.listdir(file_path)]

    documents=[]

    for book in books_file:

        loader = TextLoader(os.path.join(file_path, book))
        documents = loader.load()

        for doc in documents:
            doc.metadata = {"source": book}
            documents.append(doc)

    text_spliter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = text_spliter.split_documents(documents)

    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)

db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)
query = "Tell me about karnatka climate"

retriver = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4}
)

relavant_docs = retriver.invoke(query)

print(relavant_docs)



