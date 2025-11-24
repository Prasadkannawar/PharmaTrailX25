#!/usr/bin/env python3
"""
Clinical NLP Engine for PharmaTrail-X Phase 2
BioBERT/ClinicalBERT-powered text analysis for clinical trials
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
from loguru import logger

# NLP Libraries
import spacy
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    AutoModelForSequenceClassification, pipeline,
    BertTokenizer, BertForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class ClinicalEntity:
    """Clinical entity extracted from text"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    severity: Optional[str] = None
    category: Optional[str] = None
    context: Optional[str] = None

@dataclass
class AdverseEventExtraction:
    """Adverse event extraction result"""
    entities: List[ClinicalEntity]
    ae_events: List[Dict[str, Any]]
    severity_classification: str
    confidence_score: float
    processing_timestamp: str
    model_version: str

@dataclass
class ClinicalSummary:
    """Clinical text summarization result"""
    original_text: str
    summary: str
    key_points: List[str]
    word_count_reduction: float
    confidence_score: float
    processing_timestamp: str

class ClinicalNLPEngine:
    """
    Clinical-grade NLP engine using BioBERT/ClinicalBERT
    """
    
    def __init__(self):
        self.model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._load_models()
        
        # Clinical vocabularies and patterns
        self._init_clinical_patterns()
        
        logger.info(f"Clinical NLP Engine initialized on {self.device}")
    
    def _load_models(self):
        """Load BioBERT/ClinicalBERT models"""
        try:
            # Primary clinical BERT model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # NER pipeline for biomedical entities
            self.ner_pipeline = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                tokenizer="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load spaCy model for additional processing
            try:
                self.nlp = spacy.load("en_core_sci_sm")
            except OSError:
                logger.warning("SciSpaCy model not found, using standard English model")
                self.nlp = spacy.load("en_core_web_sm")
            
            logger.info("✅ All NLP models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to basic models
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback models if primary models fail"""
        logger.info("Loading fallback models...")
        
        # Use basic BERT model as fallback
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        self.model.to(self.device)
        
        # Basic NER pipeline
        self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
        
        # Load basic spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        logger.info("✅ Fallback models loaded")
    
    def _init_clinical_patterns(self):
        """Initialize clinical vocabulary and patterns"""
        
        # Adverse event keywords
        self.ae_keywords = {
            'mild': ['mild', 'slight', 'minor', 'minimal', 'low-grade'],
            'moderate': ['moderate', 'medium', 'noticeable', 'significant'],
            'severe': ['severe', 'serious', 'major', 'critical', 'life-threatening', 'fatal'],
            'symptoms': ['headache', 'nausea', 'fatigue', 'dizziness', 'rash', 'fever', 
                        'vomiting', 'diarrhea', 'pain', 'swelling', 'bleeding'],
            'lab_abnormal': ['elevated', 'decreased', 'abnormal', 'high', 'low', 'increased'],
            'protocol_deviation': ['deviation', 'violation', 'non-compliance', 'breach', 'error']
        }
        
        # Severity classification patterns
        self.severity_patterns = {
            'Grade 1': r'\b(grade\s*1|mild|slight|asymptomatic)\b',
            'Grade 2': r'\b(grade\s*2|moderate|symptomatic)\b',
            'Grade 3': r'\b(grade\s*3|severe|medically\s+significant)\b',
            'Grade 4': r'\b(grade\s*4|life-threatening|urgent)\b',
            'Grade 5': r'\b(grade\s*5|death|fatal|lethal)\b'
        }
        
        # Lab value patterns
        self.lab_patterns = {
            'ALT': r'\b(ALT|alanine\s+aminotransferase)\s*:?\s*(\d+\.?\d*)\s*(U/L|IU/L)?\b',
            'AST': r'\b(AST|aspartate\s+aminotransferase)\s*:?\s*(\d+\.?\d*)\s*(U/L|IU/L)?\b',
            'Creatinine': r'\b(creatinine)\s*:?\s*(\d+\.?\d*)\s*(mg/dL|μmol/L)?\b',
            'Hemoglobin': r'\b(hemoglobin|Hgb|Hb)\s*:?\s*(\d+\.?\d*)\s*(g/dL|g/L)?\b'
        }
    
    def extract_adverse_events(self, text: str, trial_id: str = None) -> AdverseEventExtraction:
        """
        Extract adverse events from clinical text
        """
        logger.info(f"Extracting adverse events from text (length: {len(text)})")
        
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Extract entities using NER
            entities = self._extract_clinical_entities(cleaned_text)
            
            # Identify adverse events specifically
            ae_events = self._identify_adverse_events(cleaned_text, entities)
            
            # Classify overall severity
            severity = self._classify_severity(cleaned_text)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(entities, ae_events)
            
            result = AdverseEventExtraction(
                entities=entities,
                ae_events=ae_events,
                severity_classification=severity,
                confidence_score=confidence,
                processing_timestamp=datetime.utcnow().isoformat(),
                model_version=self.model_name
            )
            
            logger.info(f"✅ Extracted {len(ae_events)} adverse events with {len(entities)} entities")
            return result
            
        except Exception as e:
            logger.error(f"Error in adverse event extraction: {e}")
            # Return empty result with error info
            return AdverseEventExtraction(
                entities=[],
                ae_events=[],
                severity_classification="unknown",
                confidence_score=0.0,
                processing_timestamp=datetime.utcnow().isoformat(),
                model_version=f"error: {str(e)}"
            )
    
    def summarize_clinical_text(self, text: str, max_length: int = 150) -> ClinicalSummary:
        """
        Summarize clinical text using extractive and abstractive methods
        """
        logger.info(f"Summarizing clinical text (length: {len(text)})")
        
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            original_word_count = len(cleaned_text.split())
            
            # Extract key sentences using sentence ranking
            key_sentences = self._extract_key_sentences(cleaned_text, max_sentences=5)
            
            # Create extractive summary
            summary = '. '.join(key_sentences)
            
            # Extract key points
            key_points = self._extract_key_points(cleaned_text)
            
            # Calculate metrics
            summary_word_count = len(summary.split())
            reduction = (original_word_count - summary_word_count) / original_word_count
            
            # Calculate confidence based on sentence coherence
            confidence = self._calculate_summary_confidence(summary, cleaned_text)
            
            result = ClinicalSummary(
                original_text=text[:500] + "..." if len(text) > 500 else text,
                summary=summary,
                key_points=key_points,
                word_count_reduction=reduction,
                confidence_score=confidence,
                processing_timestamp=datetime.utcnow().isoformat()
            )
            
            logger.info(f"✅ Generated summary with {reduction:.1%} word reduction")
            return result
            
        except Exception as e:
            logger.error(f"Error in text summarization: {e}")
            return ClinicalSummary(
                original_text=text[:100] + "...",
                summary="Error in summarization",
                key_points=[],
                word_count_reduction=0.0,
                confidence_score=0.0,
                processing_timestamp=datetime.utcnow().isoformat()
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess clinical text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize medical abbreviations
        text = re.sub(r'\bAE\b', 'adverse event', text, flags=re.IGNORECASE)
        text = re.sub(r'\bSAE\b', 'serious adverse event', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpt\b', 'patient', text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_clinical_entities(self, text: str) -> List[ClinicalEntity]:
        """Extract clinical entities using NER"""
        entities = []
        
        try:
            # Use biomedical NER pipeline
            ner_results = self.ner_pipeline(text)
            
            for entity in ner_results:
                clinical_entity = ClinicalEntity(
                    text=entity['word'],
                    label=entity['entity_group'],
                    start=entity['start'],
                    end=entity['end'],
                    confidence=entity['score'],
                    category=self._categorize_entity(entity['entity_group'])
                )
                entities.append(clinical_entity)
            
            # Also extract using spaCy for additional entities
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['DISEASE', 'CHEMICAL', 'SYMPTOM']:
                    clinical_entity = ClinicalEntity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8,  # Default confidence for spaCy
                        category=self._categorize_entity(ent.label_)
                    )
                    entities.append(clinical_entity)
            
        except Exception as e:
            logger.warning(f"Error in entity extraction: {e}")
        
        return entities
    
    def _identify_adverse_events(self, text: str, entities: List[ClinicalEntity]) -> List[Dict[str, Any]]:
        """Identify specific adverse events from text and entities"""
        ae_events = []
        
        # Look for explicit AE mentions
        ae_patterns = [
            r'adverse\s+event[s]?:?\s*([^.]+)',
            r'side\s+effect[s]?:?\s*([^.]+)',
            r'reaction[s]?:?\s*([^.]+)',
            r'experienced\s+([^.]+)',
            r'developed\s+([^.]+)',
            r'complained\s+of\s+([^.]+)'
        ]
        
        for pattern in ae_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ae_text = match.group(1).strip()
                
                # Extract severity if present
                severity = self._extract_severity_from_text(ae_text)
                
                ae_event = {
                    'event_text': ae_text,
                    'severity': severity,
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'confidence': 0.8,
                    'extraction_method': 'pattern_matching'
                }
                ae_events.append(ae_event)
        
        # Look for lab abnormalities
        for lab_name, pattern in self.lab_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(2)
                ae_event = {
                    'event_text': f'{lab_name} abnormality: {value}',
                    'severity': self._classify_lab_severity(lab_name, float(value) if value else 0),
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'confidence': 0.9,
                    'extraction_method': 'lab_pattern'
                }
                ae_events.append(ae_event)
        
        return ae_events
    
    def _classify_severity(self, text: str) -> str:
        """Classify overall severity of clinical text"""
        text_lower = text.lower()
        
        # Count severity indicators
        severity_scores = {
            'Grade 1': 0,
            'Grade 2': 0,
            'Grade 3': 0,
            'Grade 4': 0,
            'Grade 5': 0
        }
        
        for grade, pattern in self.severity_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            severity_scores[grade] = matches
        
        # Return highest grade found, or classify based on keywords
        max_grade = max(severity_scores.items(), key=lambda x: x[1])
        if max_grade[1] > 0:
            return max_grade[0]
        
        # Fallback classification
        if any(word in text_lower for word in self.ae_keywords['severe']):
            return 'Grade 3'
        elif any(word in text_lower for word in self.ae_keywords['moderate']):
            return 'Grade 2'
        elif any(word in text_lower for word in self.ae_keywords['mild']):
            return 'Grade 1'
        else:
            return 'Unclassified'
    
    def _extract_key_sentences(self, text: str, max_sentences: int = 5) -> List[str]:
        """Extract key sentences for summarization"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return sentences
        
        # Score sentences based on clinical importance
        sentence_scores = []
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Higher score for sentences with clinical keywords
            for keyword_list in self.ae_keywords.values():
                score += sum(1 for keyword in keyword_list if keyword in sentence_lower)
            
            # Higher score for sentences with numbers (likely lab values)
            score += len(re.findall(r'\d+\.?\d*', sentence)) * 0.5
            
            # Higher score for sentences with medical entities
            doc = self.nlp(sentence)
            score += len([ent for ent in doc.ents if ent.label_ in ['DISEASE', 'CHEMICAL']]) * 2
            
            sentence_scores.append((sentence, score))
        
        # Sort by score and return top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return [sent[0] for sent in sentence_scores[:max_sentences]]
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from clinical text"""
        key_points = []
        
        # Extract sentences with specific clinical indicators
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Key point indicators
            if any(indicator in sentence_lower for indicator in [
                'adverse event', 'side effect', 'reaction', 'abnormal', 'elevated',
                'decreased', 'protocol deviation', 'serious', 'severe'
            ]):
                key_points.append(sentence.strip())
        
        return key_points[:10]  # Limit to top 10 key points
    
    def _categorize_entity(self, entity_label: str) -> str:
        """Categorize clinical entities"""
        category_mapping = {
            'DISEASE': 'medical_condition',
            'CHEMICAL': 'medication',
            'SYMPTOM': 'adverse_event',
            'PERSON': 'patient_reference',
            'DATE': 'temporal',
            'QUANTITY': 'measurement'
        }
        return category_mapping.get(entity_label, 'other')
    
    def _extract_severity_from_text(self, text: str) -> str:
        """Extract severity classification from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in self.ae_keywords['severe']):
            return 'severe'
        elif any(word in text_lower for word in self.ae_keywords['moderate']):
            return 'moderate'
        elif any(word in text_lower for word in self.ae_keywords['mild']):
            return 'mild'
        else:
            return 'unspecified'
    
    def _classify_lab_severity(self, lab_name: str, value: float) -> str:
        """Classify lab value severity based on normal ranges"""
        # Simplified normal ranges (would be more comprehensive in production)
        normal_ranges = {
            'ALT': (7, 40),
            'AST': (8, 40),
            'Creatinine': (0.6, 1.2),
            'Hemoglobin': (12, 16)
        }
        
        if lab_name not in normal_ranges:
            return 'unspecified'
        
        min_normal, max_normal = normal_ranges[lab_name]
        
        if value > max_normal * 3:
            return 'severe'
        elif value > max_normal * 1.5:
            return 'moderate'
        elif value > max_normal or value < min_normal * 0.8:
            return 'mild'
        else:
            return 'normal'
    
    def _calculate_confidence(self, entities: List[ClinicalEntity], ae_events: List[Dict]) -> float:
        """Calculate overall confidence score"""
        if not entities and not ae_events:
            return 0.0
        
        # Average entity confidence
        entity_confidence = np.mean([e.confidence for e in entities]) if entities else 0.0
        
        # AE extraction confidence
        ae_confidence = np.mean([ae['confidence'] for ae in ae_events]) if ae_events else 0.0
        
        # Weighted average
        total_items = len(entities) + len(ae_events)
        if total_items == 0:
            return 0.0
        
        weighted_confidence = (
            (len(entities) * entity_confidence + len(ae_events) * ae_confidence) / total_items
        )
        
        return min(weighted_confidence, 1.0)
    
    def _calculate_summary_confidence(self, summary: str, original_text: str) -> float:
        """Calculate confidence score for text summarization"""
        try:
            # Use sentence similarity to measure how well summary represents original
            summary_embedding = self.sentence_model.encode([summary])
            original_embedding = self.sentence_model.encode([original_text[:1000]])  # Limit length
            
            # Calculate cosine similarity
            similarity = np.dot(summary_embedding[0], original_embedding[0]) / (
                np.linalg.norm(summary_embedding[0]) * np.linalg.norm(original_embedding[0])
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating summary confidence: {e}")
            return 0.7  # Default confidence
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'primary_model': self.model_name,
            'device': str(self.device),
            'models_loaded': {
                'clinical_bert': self.model is not None,
                'ner_pipeline': self.ner_pipeline is not None,
                'sentence_transformer': self.sentence_model is not None,
                'spacy_model': self.nlp is not None
            },
            'capabilities': [
                'adverse_event_extraction',
                'clinical_entity_recognition',
                'severity_classification',
                'text_summarization',
                'lab_value_extraction'
            ]
        }
