from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7,max_output_tokens=10)

# Simple text invocation
result = llm.invoke("Who is the PM of India?")
print(result)