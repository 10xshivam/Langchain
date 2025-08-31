from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",dimension=32)

# vector = embeddings.embed_query("Delhi is the capital of India.")
# print(str(vector))

docs = [
    "What is the capital of India?",
    "Who is the Prime Minister of India?",
    "What is the currency of India?"
]
vectors = embeddings.embed_documents(docs)
print(str(vectors))
