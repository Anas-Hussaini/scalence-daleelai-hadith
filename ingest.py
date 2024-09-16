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
    tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
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
        model_name='text-embedding-3-small'
    )
    logging.info("OpenAI embedding function initialized successfully.")
except Exception as e:
    logging.error("Failed to initialize OpenAI embedding function.", exc_info=True)
    raise e

# Create or get an existing collection for storing the hadith data
try:
    collection_name="eight_hadith_books"
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






















# print(len(documents_b_m),len(metadatas_b_m),len(ids_b_m))
# len(documents_b_m[13600:13750])
# 274*50

# # Get indices of blank values (None, empty strings, whitespace-only strings)
# blank_indices = [
#     index for index, item in enumerate(documents_b_m)
#     if item is None or (isinstance(item, str) and item.strip() == "")
# ]

# print(blank_indices)  # Output: [1, 3, 4]

# indices = [533, 696, 697, 834, 1484, 1651, 1656, 1659, 1662, 1666, 1831, 1835]


# # Using list comprehension
# selected_elements = [metadatas_b_m[i] for i in indices]
# # metadatas_b_m[23280]

# hadith_533="And he narrated to me from Malik, from Yahya ibn Said, that the Messenger of Allah (peace be upon him) was shrouded in three pure white garments."
# hadith_696="Malik said, 'I have heard the same from Sulayman ibn Yasar.' Malik continued, 'Whoever dies with an unfulfilled vow to free a slave, fast, give charity, or offer a sacrificial animal, and has bequeathed that it should be fulfilled from his estate, then the charity or the sacrificial animal is to be taken from one-third of his estate. This is given preference over other bequests, except those of a similar nature, because what is required by a vow is not like voluntary donations. Such requirements are settled from one-third of the estate specifically, rather than from the entirety of it. This is because if it were allowed to be fulfilled from the entire estate, the deceased might delay fulfilling what was obligatory upon him until death approached, and then bequeath such matters that no one would claim, as with debts. If this were permissible, he might postpone these obligations until death, bequeath them, and they might consume the entirety of his estate. Such a practice is not allowed.'"
# hadith_697="Yahya related to me from Malik that he had heard that Abdullah ibn Umar used to be asked, 'Can someone fast on behalf of someone else, or perform prayer on behalf of someone else?' and he would reply, 'No one can fast on behalf of someone else, nor can anyone pray on behalf of someone else.'"
# hadith_834="It was narrated to me from Malik that he had heard that the Messenger of Allah (peace be upon him) and his companions were in a state of ihram at Hudaybiyyah. They slaughtered their sacrificial animals, shaved their heads, and removed all restrictions before performing the Tawaf of the Kaaba and before the sacrificial animals reached their destination. It is not known that the Messenger of Allah (peace be upon him) instructed any of his companions or those with him to complete any specific acts or to return to any previous state."
# hadith_1484="""Malik related to me that he heard Ibn Shihab being asked about that and he said the like of what Sulayman ibn Yasar said. Malik said, "That is what is done in our community. It is by the word of Allah, the Blessed, the Exalted, 'And those who accuse women who are muhsan, and then do not bring four witnesses, flog them with eighty lashes, and do not accept any testimony of theirs ever. They indeed are evil-doers, save those who turn in tawba after that and make amends. Allah is Forgiving, Merciful.' " (Sura 24 ayat 4). Malik said, 'The matter on which there is no disagreement among us is that if someone is flogged for the legal punishment and then repents and reforms, his testimony is acceptable. This is the most preferred opinion to me regarding this issue.'"""
# hadith_1651="The Chapter on the Legal Punishment for Alcohol"
# hadith_1656="The Chapter on What is Prohibited to Be Used for Brewing"
# hadith_1659="The Chapter on What is Disliked to Brew All at Once"
# hadith_1662="The Chapter on the Prohibition of Alcohol"
# hadith_1666="The Chapter on the Comprehensive Prohibition of Alcohol"
# hadith_1831="""Yahya related to me from Malik from Yahya ibn Said that Umar ibn al-Khattab said, "Beware of meat. It has addictiveness like the addictiveness of wine."""
# hadith_1835="The Chapter on What Has Been Reported Regarding Removing Amulets and Bells from the Neck"

# # Indices and their new values
# indices_to_replace = [533, 696, 697, 834, 1484, 1651, 1656, 1659, 1662, 1666, 1831, 1835]
# new_values = [hadith_533,hadith_696,hadith_697,hadith_834,hadith_1484,hadith_1651,hadith_1656,hadith_1659,hadith_1662,hadith_1666,hadith_1831,hadith_1835]

# # Replace values
# for index, new_value in zip(indices_to_replace, new_values):
#     if 0 <= index < len(documents_b_m):  # Ensure index is within bounds
#         documents_b_m[index] = new_value

# print(documents_b_m)  # Output: [10, 99, 30, 88, 50]

# single_line = hadith_23280.replace("\n", " ")

# print(single_line)

# import pandas as pd
# df = pd.read_csv('The_Six_Books.csv')
# df.iloc[23280]
# # Modify a specific cell
# df.at[23280, 'Hadith Text'] = single_line

# # Save the modified DataFrame back to the CSV file
# df.to_csv('The_Six_Books_updated.csv', index=False, encoding='utf-8-sig')

# import pandas as pd

# # Read the CSV file into a DataFrame
# df = pd.read_csv('CSVs/malikDotCom.csv')

# # Define the new list of values
# new_list = documents_b_m  # Make sure this list has the same number of elements as the number of rows in the DataFrame

# # Replace the column with the new list
# df['Hadith Text'] = new_list  # Replace 'column_name' with the name of the column you want to replace

# # Save the updated DataFrame back to a CSV file
# df.to_csv('CSVs/malikDotCom_updated.csv', index=False)
