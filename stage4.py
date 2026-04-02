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

# -------------------------------
# TOOL 1 — Distance from the Sun
# -------------------------------

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

# ------------------------------------
# TOOL 2 — Revolution Period Tool
# ------------------------------------

# Expose this function as a tool named "PlanetRevolutionPeriod"
# This allows the LLM to see and call this function
@tool("PlanetRevolutionPeriod")

# Define a function named planet_revolution_period.
# It accepts one argument: planet_name (must be a string).
# It returns a string.
def planet_revolution_period(planet_name: str) -> str:
    
    # This docstring explains what the tool does.
    # The LLM reads this description to decide when to use it.
    """Returns the approximate revolution period of a planet in Earth years."""
   
    # Create a dictionary that maps planet names to their revolution periods
    periods = {
        # If the planet is Earth, return 1 Earth year.
        "Earth": "Earth takes approximately 1 Earth year to revolve around the Sun.",
        # If the planet is Mars, return 1.88 Earth years.
        "Mars": "Mars takes approximately 1.88 Earth years to revolve around the Sun.",
        # If the planet is Jupiter, return 11.86 Earth years.
        "Jupiter": "Jupiter takes approximately 11.86 Earth years to revolve around the Sun.",
        # If the planet is Pluto return 248 Earth years.
        "Pluto": "Pluto takes approximately 248 Earth years to revolve around the Sun."
    }

    # Look up the given planet name inside the dictionary.
    # If found -> return the stored revolution period.
    # If not found -> return the fallback message.
    return periods.get(
        planet_name,
        f"Information about the revolution period of {planet_name} is not available in this tool."
    ) 

# ------------------------------------
# TOOL 3 — General Planet Information
# ------------------------------------

# Expose this function as a tool named "PlanetGeneral Info"
@tool("PlanetGeneralInfo")

# Define a function named planet_general_info
# It accepts one argument: question (must be a string)
# It returns a string
def planet_general_info(question: str) -> str:
    
    # This description tells the LLM when to use this tool.
    # It will use this for general knowledge questions.
    """Searches the planet knowledge base and returns relevant information."""

    # Perform similarity search in vector database.i
    # question = user query
    # k=1 = return top 1 most relevant document

    results = vectorestore.similarity_Search(question, k=1)

    # If results found, return the page content.
    if results:
        return results[0].page_content

    # Fall back message if nothing found
    return "No relevant information found in the planet knowledge base."
































