import os
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def download_instagram_video_mp3(post_url, output_dir='audios'):
    logging.info("Starting the download process...")
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory {output_dir}")
    
    try:
        # Create an instance of Instaloader
        L = Instaloader(download_videos=True)
        logging.info("Instaloader instance created.")
        
        # Load session file
        L.load_session_from_file("insight_inn") 
        logging.info("Session loaded successfully.")
        
        # Extract the shortcode from the URL
        shortcode = post_url.split("/")[-2]
        video_filename = f"{output_dir}/{shortcode}.mp4"
        audio_filename = f"{output_dir}/{shortcode}.mp3"

        # Skip download if the MP3 file already exists
        if os.path.exists(audio_filename):
            logging.info("Audio file already exists. Skipping download.")
            return audio_filename
        
        # Download the post using the shortcode
        post = Post.from_shortcode(L.context, shortcode)
        video_url = post.video_url
    
        # Send an HTTP GET request to the video URL
        response = requests.get(video_url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the video file
            with open(video_filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            logging.info(f'Video downloaded to {video_filename}')
            
            # Convert the video file to MP3
            logging.info("Converting video to MP3...")
            video = AudioSegment.from_file(video_filename, format="mp4")
            video.export(audio_filename, format="mp3")
            logging.info(f'Audio saved as {audio_filename}')
            
            # Optionally, delete the video file after conversion
            os.remove(video_filename)
            logging.info(f'Removed temporary video file {video_filename}')
            
            return audio_filename
        else:
            logging.error(f'Failed to download video. HTTP status code: {response.status_code}')
    except Exception as e:
        logging.error(f'An error occurred in downloading or converting the video: {e}', exc_info=True)

# # Example usage
# instagram_post_url = 'https://www.instagram.com/reel/C_Qd8Tatpj_/?utm_source=ig_web_copy_link'
# output_directory = "insta_video_data"

# download_instagram_video_requests(instagram_post_url, output_directory)



# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Configure logging to save all logs in one file
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Set the log format
    filename='app.log',  # Log file name
    filemode='a'  # 'w' for overwrite, 'a' for append
)

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

        # Open the audio file
        audio_path = f"audios/{audio_filename}.mp3"
        with open(audio_path, "rb") as audio_file:
            logging.info(f"Opened audio file: {audio_path}")

            # Transcribe the audio file using Whisper
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
            hadith = transcription.text
            logging.info("Transcription completed successfully.")

        # Initialize ChromaDB client
        vectorstore_client = chromadb.PersistentClient(path=chromadb_path)
        logging.info("ChromaDB client initialized successfully.")

        # Prepare the embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_token,
            model_name="text-embedding-3-small"
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

# # Example usage
# process_audio(
#     openai_token=os.environ["OpenAI_TOKEN"],
#     audio_filename="C_XxasBNdt7",
#     chromadb_path="chroma",
#     collection_name="bukhari_muslim_collection"
# )
