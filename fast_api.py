# Import Libraries
from fastapi import FastAPI, HTTPException, WebSocket, File, UploadFile
from processes import download_instagram_video_mp3,transcript,llm_layer,query,output_parse
from pydantic import BaseModel
from fastapi.responses import FileResponse
from docx import Document

class UrlRequest(BaseModel):
    url: str
    
class TextRequest(BaseModel):
    text: str

# Create an instance of FAST API, named as "app"
app = FastAPI()


@app.get('/')
async def root():
    
    """
    Root endpoint to check if the API is running.
    """
    return {
        'message': 'This is an FAST API Daleel API trial',
        'data': 14000
    }

# Send the url and retrieve the answer
@app.post("/url-retrieve/")
async def retrieve_answer(request: UrlRequest):
    try:
        audio_filename = download_instagram_video_mp3(request.url)
        filename = audio_filename.split('/')[-1].split('.')[0]
        transcription = transcript(filename)
        hadith_crux = llm_layer(transcription)
        retrieved_docs_llm_layer = query(hadith_crux)
    
        return {
            "url": request.url,
            "message": "Retrieved successfully",
            "transcription": transcription,
            "hadith_crux": hadith_crux,
            "retrieved_docs_llm_layer": retrieved_docs_llm_layer
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve the answer: {str(e)}"
        )

# Send the text and retrieve the answer
@app.post("/text-retrieve/")
async def retrieve_answer(request: TextRequest):
    try:
        transcription = request.text
        hadith_crux = llm_layer(transcription)
        retrieved_docs_llm_layer = query(hadith_crux)
        return {
            "text": request.text,
            "message": "Retrieved successfully",
            "hadith_crux": hadith_crux,
            "retrieved_docs_llm_layer": retrieved_docs_llm_layer
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve the answer: {str(e)}"
        )
        
# Send the audio and retrieve the answer
@app.post("/audio-retrieve/")
async def retrieve_answer(file: UploadFile = File(...)):
    try:
        # Save the uploaded file or process it
        audio_filename = file.filename.split('/')[-1].split('.')[0]
        file_location = f"audios/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        transcription = transcript(audio_filename)
        hadith_crux = llm_layer(transcription)
        retrieved_docs_llm_layer = query(hadith_crux)
        return {
            "audio_filename": file.filename,
            "message": "Retrieved successfully",
            "transcription": transcription,
            "hadith_crux": hadith_crux,
            "retrieved_docs_llm_layer": retrieved_docs_llm_layer
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve the answer: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

