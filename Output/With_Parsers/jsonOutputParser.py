from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Normal way
# prompt = template.invoke({'topic':'black hole'})

# result = model.invoke(prompt)

# parsed_result = parser.parse(result.content)

# print(parsed_result)

# Chained way
chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)

# Here LLMs decides the output format but we define it as well in a Structured Output Parser
