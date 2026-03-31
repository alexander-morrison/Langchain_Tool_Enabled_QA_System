import os
import dotenv
from langcore_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import dotenv
import os

dotenv.load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
