# Import Libraries
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from processes import download_instagram_video_mp3,transcript,llm_layer,query,output_parse
from pydantic import BaseModel
from config import output_dir

class UrlRequest(BaseModel):
    url: str
    
class TextRequest(BaseModel):
    text: str

# Configure logging to write to a file and set level to INFO
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log file name
    filemode='a'  # 'w' for overwrite each time, 'a' for append
)

# If you also want to log to the console, add a console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Create an instance of FAST API, named as "app"
app = FastAPI()


@app.get('/')
async def root():
    
    """
    Root endpoint to check if the API is running.
    """
    logging.info("Root endpoint accessed")
    return {
        'message': 'This is an FAST API Daleel API trial',
        'data': 14000
    }

# Send the url and retrieve the answer
@app.post("/url-retrieve/")
async def retrieve_answer(request: UrlRequest):
    try:
        logging.info(f"Received URL request: {request.url}")
        audio_filename = download_instagram_video_mp3(request.url)
        filename = audio_filename.split('/')[-1].split('.')[0]
        transcription = transcript(filename)
        hadith_crux = llm_layer(transcription)
        retrieved_docs_llm_layer = query(hadith_crux)
    
        logging.info(f"Successfully retrieved docs for URL: {request.url}")

        return {
            "url": request.url,
            "message": "Retrieved successfully",
            "transcription": transcription,
            "hadith_crux": hadith_crux,
            "retrieved_docs_llm_layer": retrieved_docs_llm_layer
        }
    
    except Exception as e:
        logging.error(f"Error retrieving docs for URL {request.url}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve the answer: {str(e)}"
        )

# Send the text and retrieve the answer
@app.post("/text-retrieve/")
async def retrieve_answer(request: TextRequest):
    try:
        logging.info(f"Received text request: {request.text}")
        transcription = request.text
        hadith_crux = llm_layer(transcription)
        retrieved_docs_llm_layer = query(hadith_crux)
        
        logging.info(f"Successfully retrieved docs for text: {request.text}")
        
        return {
            "text": request.text,
            "message": "Retrieved successfully",
            "hadith_crux": hadith_crux,
            "retrieved_docs_llm_layer": retrieved_docs_llm_layer
        }
    except Exception as e:
        logging.error(f"Error retrieving docs for text {request.text}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve the answer: {str(e)}"
        )
        
# Send the audio and retrieve the answer
@app.post("/audio-retrieve/")
async def retrieve_answer(file: UploadFile = File(...)):
    try:
        logging.info(f"Received audio file: {file.filename}")
        # Save the uploaded file or process it
        audio_filename = file.filename.split('/')[-1].split('.')[0]
        file_location = f"{output_dir}/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        transcription = transcript(audio_filename)
        hadith_crux = llm_layer(transcription)
        retrieved_docs_llm_layer = query(hadith_crux)
        
        logging.info(f"Successfully retrieved docs for audio file: {file.filename}")
        
        return {
            "audio_filename": file.filename,
            "message": "Retrieved successfully",
            "transcription": transcription,
            "hadith_crux": hadith_crux,
            "retrieved_docs_llm_layer": retrieved_docs_llm_layer
        }
        
    except Exception as e:
        logging.error(f"Error retrieving docs for audio file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve the answer: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

