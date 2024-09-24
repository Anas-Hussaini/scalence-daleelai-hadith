import logging
from instaloader import Instaloader, Post
import requests
from pydub import AudioSegment
import os
import logging
import json
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

def process_audio(
        openai_token,
        audio_filename,
        chromadb_path,
        collection_name,
        output_dir="reports"
    ):
    """
    Processes an audio file: transcribes it using OpenAI's Whisper API, embeds the transcription, 
    and stores results in ChromaDB.
    
    Parameters:
    - openai_token (str): The API token for OpenAI.
    - audio_filename (str): The filename of the audio file to be processed (without extension).
    - chromadb_path (str): The file path to the ChromaDB storage.
    - collection_name (str): The name of the collection in ChromaDB.
    - output_dir (str): The directory where the output report should be saved.
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_token)
        logging.info("OpenAI client initialized successfully.")

        # # Open the audio file
        # audio_path = f"audios/{audio_filename}.mp3"
        # with open(audio_path, "rb") as audio_file:
        #     logging.info(f"Opened audio file: {audio_path}")

        #     # Transcribe the audio file using Whisper
        #     transcription = client.audio.transcriptions.create(
        #         model="whisper-1",
        #         file=audio_file,
        #         language="en"
        #     )
        #     hadith = transcription.text
        #     logging.info("Transcription completed successfully.")
        hadith = """The Prophet ﷺ said whenever a person makes dua, Allah answers it in three different ways. You ask Allah for something, and He gives it to you immediately. You ask Allah for something, but instead of granting it, He prevents a harm that was pre-decreed for you. You ask Allah for something, and He stores the reward for you on the Day of Judgment."""
        
        # Initialize ChromaDB client
        vectorstore_client = chromadb.PersistentClient(path=chromadb_path)
        logging.info("ChromaDB client initialized successfully.")

        # Prepare the embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_token,
            model_name="text-embedding-3-large"
        )
        logging.info("Embedding function prepared successfully.")

        # Set up the collection in ChromaDB
        collection = vectorstore_client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        logging.info(f"ChromaDB collection '{collection_name}' set up successfully.")

        # Query the collection with the transcribed text
        retrieved_docs = collection.query(query_texts=hadith, n_results=3)
        logging.info("Query executed successfully.")

        # Extract the results
        data = retrieved_docs
        metadatas = data['metadatas'][0]
        distances = data['distances'][0]
        documents = data['documents'][0]

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")

        # Save the results to a text file
        output_file_path = f"{output_dir}/results_{audio_filename}.txt"
        with open(output_file_path, "w", encoding='utf-8-sig') as file:
            file.write("Transcription:\n")
            file.write(f"{hadith}\n")
            file.write("\n" + "-"*50 + "\n\n")
            for i in range(len(documents)):
                file.write(f"Document {i+1}:\n")
                file.write(f"{documents[i]}\n")
                file.write(f"Metadata: {json.dumps(metadatas[i], indent=4)}\n")
                file.write(f"Distance: {distances[i]}\n")
                file.write("\n" + "-"*50 + "\n\n")
        logging.info(f"Results saved successfully to {output_file_path}.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        
process_audio(
    openai_token=os.environ["OpenAI_TOKEN"],
    audio_filename="C_QZ7tBNWoL.mp3",
    chromadb_path="chroma",
    collection_name="eight_hadith_books_large"
)