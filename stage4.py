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

# Load environment variables (including OpenAI_API_KEUY)
dotenv.load_dotenv()

# Create embedding model
embeddings = OpenAIEmbeddings()

# Create a loader to read all .txt files from the "planets" directory
loader = DirectoryLoader(
    # folder containing planet text files
    "planets",
    # only include .txt files
    glob="*.txt",
    # use TextLoader to read each file
    loader_cls=TextLoader
)

# Load all documents from the planets directory.
documents = loader.load()

# Create a Chroma vector store from the loaded documents
vectorstore = Chroma.from_documents(
    # the planet text documents
    documents,
    # the embedding model used to convert them into vectors
    embeddings
)

# Expose this function as a tool the LLM can use.
@tool("PlanetDistanceSun")

# Define the function that the LLM can call.
# It must accept one argument: planet_name (string)
# It must return a string.
def planet_distance_sun(planet_name: str) -> str: 
    # This docstring describes what the tool does.
    # The LLM reads this description to decide when to use this tool.
    """Return the approximate distance of a planet from the Sun in AU."""
    
    # Dictionary mapping planet names to their predefined distance.
    distances = {
        "Earth": "Earth is approximately 1 AU from the Sun.",
        "Mars": "Mars is approximately 1.5 AU from the Sun.",
        "Jupiter": "Jupiter is approximately 5.2 AU from the Sun.",
        "Pluto": "Pluto is approximately 39.5 AU from the Sun." 
    }

    # Look up the planet name in the dictionary.
    # If found -> return its distance.
    # if not found -> return the fallback message.
    return distances.get(
        planet_name,
        f"Information about the distance of {planet_name} from the Sun is not available in this tool."
    )

        




















