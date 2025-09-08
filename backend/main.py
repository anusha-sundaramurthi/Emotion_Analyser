from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import logging
import uvicorn
from nlp import SemanticEmotionAnalyzer

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion Analyzer API",
    description="AI-Powered Emotion Detection",
    version="1.0.0"
)

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class TextAnalysisRequest(BaseModel):
    text: str

class EmotionScore(BaseModel):
    joy: int = 0
    sadness: int = 0
    anger: int = 0
    fear: int = 0
    surprise: int = 0
    disgust: int = 0
    neutral: int = 0

class EmotionInfo(BaseModel):
    emoji: str
    color: str
    name: str

class AnalysisResponse(BaseModel):
    emotion: EmotionInfo
    emotion_breakdown: EmotionScore


# Initialize analyzer
logger.info("Initializing Emotion Analyzer...")
analyzer = SemanticEmotionAnalyzer()


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Emotion Analyzer API",
        "version": "1.0.0",
        "description": "AI-Powered Emotion Detection",
        "endpoints": {
            "analyze": "/analyze - POST - Detect dominant emotion",
            "health": "/health - GET - Health check",
            "emotions": "/emotions - GET - Supported emotions",
            "docs": "/docs - GET - API documentation"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "analyzer": "ready",
        "emotions_supported": list(analyzer.emotion_patterns.keys()),
    }

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_text(request: TextAnalysisRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        result = analyzer.analyze_text(request.text)

        response = AnalysisResponse(
            emotion=EmotionInfo(
                emoji=result['emotion']['emoji'],
                color=result['emotion']['color'],
                name=result['emotion']['name']
            ),
            emotion_breakdown=EmotionScore(**result['emotion_breakdown'])
        )

        return response

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/emotions", tags=["Reference"])
async def get_supported_emotions():
    return {
        "supported_emotions": {
            emotion: {
                "emoji": data['emoji'],
                "color": data['color'],
                "examples": data['examples'][:3]
            }
            for emotion, data in analyzer.emotion_patterns.items()
        }
    }


if __name__ == "__main__":
    print("🚀 Starting FastAPI Emotion Analyzer Backend...")

