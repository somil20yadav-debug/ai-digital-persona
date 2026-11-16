"""Database management for storing samples and conversations"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import config

Base = declarative_base()

class WritingSample(Base):
    __tablename__ = "writing_samples"
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    source = Column(String(200))
    added_at = Column(DateTime, default=datetime.utcnow)
    word_count = Column(Integer)
    
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    similarity_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    persona_name = Column(String(100), default="default")

class StyleMetrics(Base):
    __tablename__ = "style_metrics"
    
    id = Column(Integer, primary_key=True)
    avg_sentence_length = Column(Float)
    vocab_richness = Column(Float)
    formality_score = Column(Float)
    readability_score = Column(Float)
    common_phrases = Column(Text)  # JSON string
    updated_at = Column(DateTime, default=datetime.utcnow)

# Database engine
engine = create_engine(config.DATABASE_URL)
Base.metadata.create_all(engine)

# 🛠️ KEY FIX: Objects won't detach after commit
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_sample(content: str, source: str = "manual"):
    db = SessionLocal()
    sample = WritingSample(
        content=content,
        source=source,
        word_count=len(content.split())
    )
    db.add(sample)
    db.commit()
    sample_id = sample.id  # Safe because expire_on_commit=False
    db.close()
    return sample_id

def get_all_samples():
    db = SessionLocal()
    samples = db.query(WritingSample).all()
    db.close()
    return samples

def save_conversation(prompt: str, response: str, similarity: float, persona: str = "default"):
    db = SessionLocal()
    conv = Conversation(
        prompt=prompt,
        response=response,
        similarity_score=similarity,
        persona_name=persona
    )
    db.add(conv)
    db.commit()
    db.close()

def get_conversations(limit: int = 50):
    db = SessionLocal()
    convs = db.query(Conversation).order_by(Conversation.timestamp.desc()).limit(limit).all()
    db.close()
    return convs

def save_style_metrics(metrics: dict):
    import json
    db = SessionLocal()
    style = StyleMetrics(
        avg_sentence_length=metrics.get('avg_sentence_length', 0),
        vocab_richness=metrics.get('vocab_richness', 0),
        formality_score=metrics.get('formality_score', 0),
        readability_score=metrics.get('readability_score', 0),
        common_phrases=json.dumps(metrics.get('common_phrases', []))
    )
    db.add(style)
    db.commit()
    db.close()

def get_latest_metrics():
    db = SessionLocal()
    metrics = db.query(StyleMetrics).order_by(StyleMetrics.updated_at.desc()).first()
    db.close()
    return metrics

def delete_all_data():
    """Delete all samples, conversations, and style metrics."""
    db = SessionLocal()
    db.query(WritingSample).delete()
    db.query(Conversation).delete()
    db.query(StyleMetrics).delete()
    db.commit()
    db.close()
