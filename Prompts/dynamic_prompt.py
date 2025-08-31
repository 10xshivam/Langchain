from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv

load_dotenv()

template = load_prompt("Prompts/prompt_template.json")

prompt = template.invoke({
    'paper_input': "Attention is All You Need",
    'style_input': "Beginner friendly",
    'length_input': "Short (1-2 paragraphs)"
})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Simple text invocation
result = llm.invoke(prompt)
print(result.content)

