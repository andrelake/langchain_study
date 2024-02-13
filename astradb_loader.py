import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAIEmbeddings

# Load env variables
load_dotenv()

# Get OpenAI Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Get AstraDB Keys
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_KEYSPACE = os.getenv('ASTRA_DB_KEYSPACE')
ASTRA_DB_COLLECTION_NAME = os.getenv('TABLE_NAME')


def get_vector_store():
    # Set OpenAI Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Build vectorstore
    vectorstore = AstraDB(
        embedding=embeddings,
        collection_name=ASTRA_DB_COLLECTION_NAME,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )
    return vectorstore


def load_data(vectorstore):
    # Get content from file
    loader = DirectoryLoader('data', glob="**/*.txt")
    docs = loader.load()
    return vectorstore.add_documents(docs)


def main() -> None:
    # Get vectorstore
    vectorstore = get_vector_store()
    # Persist data into vector database
    data = load_data(vectorstore)


if __name__ == "__main__":
    main()
