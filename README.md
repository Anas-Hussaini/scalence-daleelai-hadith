# **Instagram Reels Hadith Finder**

This project processes Instagram Reels from a given link, extracts and analyzes the audio, and outputs the **three most similar Ahadith** based on the content. The project utilizes **OpenAI**, **ChromaDB**, and **Whisper 1** to achieve accurate and insightful results.

## **Features**
- Extracts audio content from Instagram Reels using Whisper 1 model.
- Transcribes the audio into text using **OpenAI’s Whisper**.
- Analyzes the transcribed content using **OpenAI’s language model** for semantic understanding.
- Finds and returns the **three most semantically similar Ahadith** to the content using **ChromaDB** for vector storage and retrieval.

## **Tech Stack**
- **OpenAI GPT**: For processing and analyzing the content of the transcribed audio.
- **Whisper 1**: For transcribing audio from Instagram Reels into text.
- **ChromaDB**: To store and retrieve the Ahadith embeddings for similarity comparison.
- **Instagram API**: For fetching Instagram Reel data via link.

## **Requirements**
To run this project, you need the following dependencies installed:
- **Python 3.8+**
- **OpenAI GPT API Key**: To interact with OpenAI’s models.
- **Instagram API Access**: To fetch reel data.
- **ChromaDB**: For managing vector embeddings.
- **Whisper Model**: For audio transcription.

### **Python Libraries**:
requirements.txt

You can install the required libraries using the following command:
```bash
pip install -r requirements.txt
```

How It Works
```
Input: The project accepts an Instagram Reel link.
Audio Extraction: Using Whisper, the audio from the Reel is extracted and transcribed into text.
Text Analysis: The transcribed text is analyzed using OpenAI’s GPT model to understand its meaning and context.
Hadith Matching: ChromaDB is used to store vector embeddings of a Hadith dataset. The project queries ChromaDB to find the three most semantically similar Ahadith to the transcribed text.
```

Set up environment variables:
```
OpenAI_TOKEN: Set up your OpenAI API key as an environment variable.
```

Future Improvements:
```
Add support for multi-language transcription and Hadith matching.
Improve Hadith database to include multiple collections.
Incorporate sentiment analysis for better understanding of the context.
```

Contributing:
```
Contributions are welcome! Please submit a pull request or open an issue to improve this project.
```

License:
```
This project is licensed under the Scalence pvt ltd.
```
