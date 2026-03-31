import os                                               
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate

# load the api
dotenv.load_dotenv()

# create and store the model
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# generate response from model
response = llm.invoke(prompt)

# print generated answer
print(response.content)
