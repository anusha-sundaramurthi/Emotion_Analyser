import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SemanticEmotionAnalyzer:
    def __init__(self):
        logger.info("Loading semantic model...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Semantic model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            self.model = None

        # Emotion patterns
        self.emotion_patterns = {
            'joy': {'examples': ["I am extremely happy", "This brings me joy"],
                    'emoji': '😊', 'color': '#2ed573'},
            'sadness': {'examples': ["I feel sad", "This makes me gloomy"],
                        'emoji': '😢', 'color': '#74b9ff'},
            'anger': {'examples': ["I am furious", "This makes me mad"],
                      'emoji': '😠', 'color': '#ff4757'},
            'fear': {'examples': ["I am terrified", "I feel anxious"],
                     'emoji': '😨', 'color': '#a29bfe'},
            'surprise': {'examples': ["I am shocked", "This is unexpected"],
                         'emoji': '😲', 'color': '#fdcb6e'},
            'disgust': {'examples': ["This is disgusting", "I feel revolted"],
                        'emoji': '🤢', 'color': '#00b894'},
            'neutral': {'examples': ["This is okay", "I feel normal"],
                        'emoji': '😐', 'color': '#ffa502'}
        }

        self._precompute_embeddings()

    def _precompute_embeddings(self):
        self.emotion_embeddings = {}
        if self.model is None:
            return
        for emotion, data in self.emotion_patterns.items():
            embeddings = self.model.encode(data['examples'])
            self.emotion_embeddings[emotion] = np.mean(embeddings, axis=0)

    def detect_emotions_semantic(self, text: str) -> Dict[str, int]:
        if self.model is None or not self.emotion_embeddings:
            return {e: 0 for e in self.emotion_patterns}

        text_embedding = self.model.encode([text])
        emotion_scores = {}
        for emotion, emb in self.emotion_embeddings.items():
            similarity = cosine_similarity(text_embedding, emb.reshape(1, -1))[0][0]
            emotion_scores[emotion] = max(0, int(similarity * 100))
        return emotion_scores

    def get_dominant_emotion(self, scores: Dict[str, int]):
        if not any(scores.values()):
            return {**self.emotion_patterns['neutral'], 'name': 'neutral'}
        top = max(scores, key=scores.get)
        return {**self.emotion_patterns[top], 'name': top}

    def analyze_text(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {
                'emotion': {**self.emotion_patterns['neutral'], 'name': 'neutral'},
                'emotion_breakdown': {e: 0 for e in self.emotion_patterns}
            }

        scores = self.detect_emotions_semantic(text)
        dominant = self.get_dominant_emotion(scores)
        total = sum(scores.values())
        breakdown = {e: int((s / total) * 100) if total > 0 else 0 for e, s in scores.items()}

        return {
            'emotion': dominant,
            'emotion_breakdown': breakdown
        }