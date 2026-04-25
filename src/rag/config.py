from os import environ as env

from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike

load_dotenv()


SILICONFLOW_BASE_URL = env.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1")

llm = OpenAILike(
    model=env.get("SILICONFLOW_MODEL", "zai-org/GLM-5V-Turbo"),
    api_base=SILICONFLOW_BASE_URL,
    api_key=env.get("SILICONFLOW_API_KEY"),
    is_chat_model=True,
    is_function_calling_model=True,
    temperature=0.3,
    max_tokens=7000,
    timeout=120,
)
