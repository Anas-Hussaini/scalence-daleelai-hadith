from functions import download_instagram_video_mp3,process_audio
import os

# Example usage
# instagram_post_url = "https://www.instagram.com/reel/C9Bw31BtkGw/?igsh=MW8yNGV2bjhvOWNyMw=="
# instagram_post_url = "https://www.instagram.com/reel/C7WYYBzNG6L/?igsh=a3BuN3VlbmhxNWRs"
# instagram_post_url = "https://www.instagram.com/reel/C79PisKI47n/?igsh=MXdyN2tmaTBsdW5iMw=="
# instagram_post_url = "https://www.instagram.com/reel/C7WogOpNrJI/?igsh=MW1vN2lud2ZveGNzag=="
# instagram_post_url = "https://www.instagram.com/reel/C_zxvJxtypA/?igsh=NWxheXVrYjVlOHVy"
# instagram_post_url = "https://www.instagram.com/reel/C2KxkCOMiNA/?igsh=MTZhb3I0ZWlxZHA2OQ=="
# instagram_post_url = "https://www.instagram.com/reel/C6Gq8vbPV1b/?igsh=MWR5OTI5b2Z5Yzdpeg=="
# instagram_post_url = "https://www.instagram.com/reel/C_l27VOP9S3/?igsh=am16c3o1ZG5rcTd5"
# instagram_post_url = "https://www.instagram.com/reel/CzuwVO9r3j1/?igsh=YXR2bG0xazIyOTc="
instagram_post_url = "https://www.instagram.com/reel/C7vZhOuvzc6/?igsh=MThiZmExY3JhYmI3bQ=="
output_directory = "audios"

# Download Instagram Video with video_url
download_instagram_video_mp3(instagram_post_url, output_directory)

# Process Audio to fetch Ahadith
audio_filename = instagram_post_url.split("/")[-2]
        
# audio_filename = 'test_bukhari_001'

process_audio(
    openai_token=os.environ["OpenAI_TOKEN"],
    audio_filename=audio_filename,
    chromadb_path="chroma",
    collection_name="eight_hadith_books"
)



# from moviepy.editor import *

# video = VideoFileClip("test_bukhari.mp4")

# # Extract the audio and save it as MP3
# video.audio.write_audiofile("test_bukhari_audio.mp3")


