import csv
import math
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import os
import tiktoken
import logging
import requests

# Configure logging to write to a file and set level to INFO
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log file name
    filemode='w'  # 'w' for overwrite each time, 'a' for append
)

# If you also want to log to the console, add a console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Load environment variables from .env file
try:
    load_dotenv(dotenv_path=".env", override=True)
    logging.info("Environment variables loaded successfully.")
except Exception as e:
    logging.error("Failed to load environment variables.", exc_info=True)
    raise e

try:
    OpenAI_TOKEN = os.environ["OpenAI_TOKEN"]
    logging.info("OpenAI token retrieved successfully.")
except KeyError as e:
    logging.error("OpenAI_TOKEN not found in environment variables.", exc_info=True)
    raise e

# Initialize lists for storing data to be inserted into the Chromadb collection
documents_b_m = []
embeddings_b_m = []
metadatas_b_m = []
ids_b_m = []

# Read CSV file containing the hadith data
try:
    with open('The_Eight_Books.csv', encoding='utf-8-sig') as file:
        logging.info("CSV file opened successfully.")
        lines = csv.reader(file)
        id = 1

        # Skip the header and process each line
        for i, line in enumerate(lines):
            if i == 0:
                logging.info("Skipping CSV header.")
                continue

            # Store hadith, metadata, and ID
            documents_b_m.append(line[0])
            metadatas_b_m.append({"hadith_id": line[3], "source": line[2]})
            ids_b_m.append(str(id))
            id += 1

    logging.info(f"Successfully loaded {len(documents_b_m)} hadiths.")
except FileNotFoundError as e:
    logging.error("CSV file not found. Please check the file path.", exc_info=True)
    raise e
except Exception as e:
    logging.error("An error occurred while reading the CSV file.", exc_info=True)
    raise e

# Initialize tokenizer for the embedding model
try:
    tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")
    logging.info("Tokenizer initialized successfully.")
except Exception as e:
    logging.error("Failed to initialize tokenizer.", exc_info=True)
    raise e

# Tokenize each hadith and log the max token length
try:
    tokens_len = [len(tokenizer.encode(text=chunk)) for chunk in documents_b_m]
    logging.info(f"Max tokens in a chunk: {max(tokens_len)}")
    logging.info(f"Number of chunks: {len(tokens_len)}")
except Exception as e:
    logging.error("Error occurred during tokenization.", exc_info=True)
    raise e

# Initialize a persistent Chromadb client
try:
    vectorstore_client = chromadb.PersistentClient(path="chroma")
    logging.info("Chromadb client initialized successfully.")
except Exception as e:
    logging.error("Failed to initialize Chromadb client.", exc_info=True)
    raise e

# List existing collections and log them
try:
    collections = vectorstore_client.list_collections()
    for col in collections:
        logging.info(f"Collection {col.name} contains {col.count()} items.")
except Exception as e:
    logging.error("Error listing collections.", exc_info=True)
    raise e

# Initialize embedding function with OpenAI API
try:
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OpenAI_TOKEN,
        model_name='text-embedding-3-large'
    )
    logging.info("OpenAI embedding function initialized successfully.")
except Exception as e:
    logging.error("Failed to initialize OpenAI embedding function.", exc_info=True)
    raise e

# Create or get an existing collection for storing the hadith data
try:
    collection_name="eight_hadith_books_large"
    collection = vectorstore_client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef,
        metadata={
            "hnsw:batch_size": 50000,
            "hnsw:sync_threshold": 1000000,
        }
    )
    logging.info(f"Chromadb collection {collection_name} created or retrieved successfully.")
except Exception as e:
    logging.error("Failed to create or retrieve Chromadb collection.", exc_info=True)
    raise e

# Define batch size and calculate the number of loops required
batch_size = 150
loops = math.ceil(len(documents_b_m) / batch_size)
logging.info(f"Data will be uploaded in {loops} batches of {batch_size} documents each.")

# Store data in the Chromadb collection in batches
for i in range(0, loops):
    try:
        logging.info(f'Starting batch upload {i+1}/{loops}')
        collection.upsert(
            documents=documents_b_m[(i * batch_size):(batch_size * (i + 1))],
            metadatas=metadatas_b_m[i * batch_size:(batch_size * (i + 1))],
            ids=ids_b_m[i * batch_size:(batch_size * (i + 1))]
        )
        logging.info(f'Successfully uploaded batch {i+1}/{loops}')
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request failed during batch {i+1}/{loops}.", exc_info=True)
        raise e
    except Exception as e:
        logging.error(f"An error occurred during batch {i+1}/{loops}.", exc_info=True)
        raise e

logging.info("Ingestion completed successfully.")
