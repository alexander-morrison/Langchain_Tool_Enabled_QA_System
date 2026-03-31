import os
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# load the api
dotenv.load_dotenv()

# create and store the model
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# user input
question = input()

# create the prompt template
template = PromptTemplate.from_template(
    "You are a helpful assistant who answers questions users may have. You are asked: {question}."
)

# Fill template with user question
prompt = template.invoke({"question": question})

# generate response from model
response = llm.invoke(prompt)

# print generated answer
print(response.content)
