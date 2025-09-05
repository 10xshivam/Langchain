from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

ytt_api = YouTubeTranscriptApi()
try:
    fetched = ytt_api.fetch("iS5tZ9WdO0k", languages=['en', 'en-US'])
    transcript = " ".join(entry["text"] for entry in fetched.to_raw_data())
    # print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video.")
    
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

# question = "What is the main topic of the video?"

# retrieved_docs = retriever.invoke(question)

# Without chains
# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# final_prompt = prompt.invoke({"context": context_text, "question": question})

# answer = llm.invoke(final_prompt)
# print(answer.content)


# With help of chains
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser
result = main_chain.invoke('What is the topic of the video?')
print(result)