from functions import download_instagram_video_mp3,process_audio
import os

output_directory = "audios"

def process_url(instagram_post_url: str):
    
    # Download Instagram Video with video_url
    download_instagram_video_mp3(instagram_post_url, output_directory)

    # Process Audio to fetch Ahadith
    audio_filename = instagram_post_url.split("/")[-2]
    # print(audio_filename)

    # Process Audio to get Relevant Ahadith
    process_audio(
        openai_token=os.environ["OpenAI_TOKEN"],
        audio_filename=audio_filename,
        chromadb_path="chroma",
        collection_name="eight_hadith_books_large"
    )
    return

    
    