# Import dotenv to load environment variables from .env file.
import dotenv

# Import OpenAI embedding model to convert text into vectors.
from langchain_openai import OpenAIEmbeddings

# Import Chroma vector database to store and search embeddings.
from langchain_community.vectorstores import Chroma

# Import tools to load all text files from a directory.
from langchain_community.document_loaders import DirectoryLoader, TextLoader

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

