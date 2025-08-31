from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/Downloads/huggingface_cache'
llm = HuggingFacePipeline.from_model_id(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=100,
        temperature=0.5,
    ),
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("Who is the PM of India?")

print(result)