from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text = "This is a test document."
result = embeddings.embed_query(text)

print(str(result))