# Import dotenv to load environment variables from .env file.

import dotenv

# Import OpenAI embedding model to convert text into vectors.
from langchain_openai import OpenAIEmbeddings

# Import Chroma vector database to store and search embeddings.
from langchain_community.vectorstores import Chroma

# Import tools to load all text files from a directory.
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Import the tool decorator to create custom tools the LLM can call.
from langchain_core.tools import tool

# Import the ChatOpenAI class to create and interact with OpenAI.
from langchain_openai import ChatOpenAI

# Import PromptTemplate to format user input before sending it to the model.
from langchain_core.prompts import PromptTemplate

# Import RunnableLambda to convert a custom Python function into a runnable
# so it can be used inside a LangChain pipeline.
from langchain_core.runnables import RunnableLambda

# Load environment variables (including OPENAI_API_KEY)
dotenv.load_dotenv()


# Create the language model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# Create embedding model
embeddings = OpenAIEmbeddings()


# Create a loader to read all .txt files from the "planets" directory
loader = DirectoryLoader(
    "planets",
    glob="*.txt",
    loader_cls=TextLoader
)


# Load all documents from the planets directory
documents = loader.load()


# Create a Chroma vector store from the loaded documents
vectorstore = Chroma.from_documents(
    documents,
    embeddings
)


# -------------------------------
# TOOL 1 — Distance from the Sun
# -------------------------------

@tool("PlanetDistanceSun")
def planet_distance_sun(planet_name: str) -> str:
    """Return the approximate distance of a planet from the Sun in AU."""

    distances = {
        "Earth": "Earth is approximately 1 AU from the Sun.",
        "Mars": "Mars is approximately 1.5 AU from the Sun.",
        "Jupiter": "Jupiter is approximately 5.2 AU from the Sun.",
        "Pluto": "Pluto is approximately 39.5 AU from the Sun."
    }

    return distances.get(
        planet_name,
        f"Information about the distance of {planet_name} from the Sun is not available in this tool."
    )


# ------------------------------------
# TOOL 2 — Revolution Period Tool
# ------------------------------------

@tool("PlanetRevolutionPeriod")
def planet_revolution_period(planet_name: str) -> str:
    """Returns the approximate revolution period of a planet in Earth years."""

    periods = {
        "Earth": "Earth takes approximately 1 Earth year to revolve around the Sun.",
        "Mars": "Mars takes approximately 1.88 Earth years to revolve around the Sun.",
        "Jupiter": "Jupiter takes approximately 11.86 Earth years to revolve around the Sun.",
        "Pluto": "Pluto takes approximately 248 Earth years to revolve around the Sun."
    }

    return periods.get(
        planet_name,
        f"Information about the revolution period of {planet_name} is not available in this tool."
    )


# ------------------------------------
# TOOL 3 — General Planet Information
# ------------------------------------

@tool("PlanetGeneralInfo")
def planet_general_info(question: str) -> str:
    """Searches the planet knowledge base and returns relevant information."""

    results = vectorstore.similarity_search(question, k=1)

    if results:
        return results[0].page_content

    return "No relevant information found in the planet knowledge base."


# Create a list containing all available tools
tools = [
    planet_distance_sun,
    planet_revolution_period,
    planet_general_info
]


# Bind tools to the model
model_with_tools = llm.bind_tools(tools)

