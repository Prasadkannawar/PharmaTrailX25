#!/usr/bin/env python3
"""
Real NLP Engine for PharmaTrail-X Phase 2
Simple but functional adverse event extraction and summarization
"""

import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from loguru import logger

@dataclass
class ExtractedEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float
    severity: Optional[str] = None

@dataclass
class AEExtractionResult:
    entities: List[ExtractedEntity]
    ae_events: List[Dict[str, Any]]
    severity_classification: str
    confidence_score: float
    processing_timestamp: str

@dataclass
class SummaryResult:
    summary: str
    key_points: List[str]
    word_count_reduction: float
    confidence_score: float
    processing_timestamp: str

class ClinicalNLPEngine:
    """
    Clinical NLP Engine for adverse event extraction and text summarization
    Uses rule-based approach with clinical dictionaries (can be upgraded to BERT later)
    """
    
    def __init__(self):
        self.ae_terms = {
            # Common adverse events with severity indicators
            'headache': {'severity': 'mild', 'category': 'neurological'},
            'severe headache': {'severity': 'severe', 'category': 'neurological'},
            'nausea': {'severity': 'mild', 'category': 'gastrointestinal'},
            'vomiting': {'severity': 'moderate', 'category': 'gastrointestinal'},
            'fatigue': {'severity': 'mild', 'category': 'general'},
            'severe fatigue': {'severity': 'severe', 'category': 'general'},
            'dizziness': {'severity': 'mild', 'category': 'neurological'},
            'hypertension': {'severity': 'moderate', 'category': 'cardiovascular'},
            'elevated blood pressure': {'severity': 'moderate', 'category': 'cardiovascular'},
            'liver enzyme elevation': {'severity': 'moderate', 'category': 'hepatic'},
            'elevated alt': {'severity': 'moderate', 'category': 'hepatic'},
            'elevated ast': {'severity': 'moderate', 'category': 'hepatic'},
            'kidney dysfunction': {'severity': 'severe', 'category': 'renal'},
            'elevated creatinine': {'severity': 'moderate', 'category': 'renal'},
            'rash': {'severity': 'mild', 'category': 'dermatological'},
            'severe rash': {'severity': 'severe', 'category': 'dermatological'},
            'chest pain': {'severity': 'severe', 'category': 'cardiovascular'},
            'shortness of breath': {'severity': 'moderate', 'category': 'respiratory'},
            'decreased appetite': {'severity': 'mild', 'category': 'gastrointestinal'},
            'weight loss': {'severity': 'moderate', 'category': 'general'},
            'insomnia': {'severity': 'mild', 'category': 'neurological'},
            'depression': {'severity': 'moderate', 'category': 'psychiatric'},
            'anxiety': {'severity': 'mild', 'category': 'psychiatric'}
        }
        
        self.lab_abnormalities = {
            'alt': {'normal_range': (0, 40), 'unit': 'U/L'},
            'ast': {'normal_range': (0, 40), 'unit': 'U/L'},
            'creatinine': {'normal_range': (0.6, 1.2), 'unit': 'mg/dL'},
            'bilirubin': {'normal_range': (0.2, 1.2), 'unit': 'mg/dL'},
            'hemoglobin': {'normal_range': (12, 16), 'unit': 'g/dL'},
            'platelet': {'normal_range': (150, 450), 'unit': '10^3/μL'}
        }
        
        self.severity_keywords = {
            'mild': ['mild', 'slight', 'minor', 'grade 1'],
            'moderate': ['moderate', 'grade 2'],
            'severe': ['severe', 'serious', 'grade 3', 'grade 4'],
            'life-threatening': ['life-threatening', 'grade 5', 'fatal']
        }
        
        logger.info("✅ Clinical NLP Engine initialized (rule-based)")
    
    def extract_adverse_events(self, text: str, trial_id: str) -> AEExtractionResult:
        """
        Extract adverse events from clinical text
        """
        try:
            text_lower = text.lower()
            entities = []
            ae_events = []
            
            # Extract AE terms
            for ae_term, ae_info in self.ae_terms.items():
                if ae_term in text_lower:
                    start_idx = text_lower.find(ae_term)
                    end_idx = start_idx + len(ae_term)
                    
                    entity = ExtractedEntity(
                        text=ae_term,
                        label="ADVERSE_EVENT",
                        start=start_idx,
                        end=end_idx,
                        confidence=0.85,
                        severity=ae_info['severity']
                    )
                    entities.append(entity)
                    
                    ae_events.append({
                        'event_text': ae_term,
                        'severity': ae_info['severity'],
                        'category': ae_info['category'],
                        'confidence': 0.85,
                        'trial_id': trial_id
                    })
            
            # Extract lab abnormalities
            lab_entities, lab_events = self._extract_lab_abnormalities(text, trial_id)
            entities.extend(lab_entities)
            ae_events.extend(lab_events)
            
            # Extract protocol deviations
            deviation_entities, deviation_events = self._extract_protocol_deviations(text, trial_id)
            entities.extend(deviation_entities)
            ae_events.extend(deviation_events)
            
            # Determine overall severity
            severity_classification = self._classify_overall_severity(ae_events)
            
            # Calculate confidence score
            confidence_score = sum(e.confidence for e in entities) / len(entities) if entities else 0.0
            
            result = AEExtractionResult(
                entities=entities,
                ae_events=ae_events,
                severity_classification=severity_classification,
                confidence_score=confidence_score,
                processing_timestamp=datetime.utcnow().isoformat()
            )
            
            logger.info(f"✅ Extracted {len(entities)} entities, {len(ae_events)} AE events")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in AE extraction: {e}")
            return AEExtractionResult(
                entities=[],
                ae_events=[],
                severity_classification="unknown",
                confidence_score=0.0,
                processing_timestamp=datetime.utcnow().isoformat()
            )
    
    def summarize_clinical_text(self, text: str, max_length: int = 150) -> SummaryResult:
        """
        Summarize clinical text and extract key points
        """
        try:
            sentences = self._split_into_sentences(text)
            
            # Score sentences based on clinical importance
            scored_sentences = []
            for sentence in sentences:
                score = self._score_sentence_importance(sentence)
                scored_sentences.append((sentence, score))
            
            # Sort by importance and select top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Build summary within max_length
            summary_sentences = []
            current_length = 0
            
            for sentence, score in scored_sentences:
                if current_length + len(sentence) <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            summary = ' '.join(summary_sentences)
            
            # Extract key points
            key_points = self._extract_key_points(text)
            
            # Calculate word count reduction
            original_words = len(text.split())
            summary_words = len(summary.split())
            word_count_reduction = (original_words - summary_words) / original_words if original_words > 0 else 0
            
            result = SummaryResult(
                summary=summary,
                key_points=key_points,
                word_count_reduction=word_count_reduction,
                confidence_score=0.75,  # Rule-based confidence
                processing_timestamp=datetime.utcnow().isoformat()
            )
            
            logger.info(f"✅ Summarized text: {original_words} -> {summary_words} words ({word_count_reduction:.1%} reduction)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in text summarization: {e}")
            return SummaryResult(
                summary=text[:max_length] + "..." if len(text) > max_length else text,
                key_points=[],
                word_count_reduction=0.0,
                confidence_score=0.0,
                processing_timestamp=datetime.utcnow().isoformat()
            )
    
    def _extract_lab_abnormalities(self, text: str, trial_id: str) -> tuple:
        """Extract lab value abnormalities"""
        entities = []
        events = []
        
        # Look for lab values with numbers
        lab_patterns = [
            r'alt\s+(\d+(?:\.\d+)?)\s*u/l',
            r'ast\s+(\d+(?:\.\d+)?)\s*u/l', 
            r'creatinine\s+(\d+(?:\.\d+)?)\s*mg/dl',
            r'bilirubin\s+(\d+(?:\.\d+)?)\s*mg/dl'
        ]
        
        text_lower = text.lower()
        
        for pattern in lab_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                lab_name = match.group(0).split()[0]
                lab_value = float(match.group(1))
                
                if lab_name in self.lab_abnormalities:
                    normal_range = self.lab_abnormalities[lab_name]['normal_range']
                    unit = self.lab_abnormalities[lab_name]['unit']
                    
                    if lab_value < normal_range[0] or lab_value > normal_range[1]:
                        severity = 'severe' if lab_value > normal_range[1] * 2 else 'moderate'
                        
                        entity = ExtractedEntity(
                            text=match.group(0),
                            label="LAB_ABNORMALITY",
                            start=match.start(),
                            end=match.end(),
                            confidence=0.90,
                            severity=severity
                        )
                        entities.append(entity)
                        
                        events.append({
                            'event_text': f'{lab_name.upper()} {lab_value} {unit}',
                            'severity': severity,
                            'category': 'laboratory',
                            'confidence': 0.90,
                            'trial_id': trial_id,
                            'lab_name': lab_name,
                            'lab_value': lab_value,
                            'normal_range': normal_range
                        })
        
        return entities, events
    
    def _extract_protocol_deviations(self, text: str, trial_id: str) -> tuple:
        """Extract protocol deviations"""
        entities = []
        events = []
        
        deviation_keywords = [
            'protocol deviation', 'deviation', 'missed dose', 'late dose',
            'inclusion criteria violation', 'exclusion criteria', 'non-compliance'
        ]
        
        text_lower = text.lower()
        
        for keyword in deviation_keywords:
            if keyword in text_lower:
                start_idx = text_lower.find(keyword)
                end_idx = start_idx + len(keyword)
                
                entity = ExtractedEntity(
                    text=keyword,
                    label="PROTOCOL_DEVIATION",
                    start=start_idx,
                    end=end_idx,
                    confidence=0.80,
                    severity='moderate'
                )
                entities.append(entity)
                
                events.append({
                    'event_text': keyword,
                    'severity': 'moderate',
                    'category': 'protocol_deviation',
                    'confidence': 0.80,
                    'trial_id': trial_id
                })
        
        return entities, events
    
    def _classify_overall_severity(self, ae_events: List[Dict]) -> str:
        """Classify overall severity based on individual events"""
        if not ae_events:
            return "none"
        
        severities = [event.get('severity', 'mild') for event in ae_events]
        
        if 'life-threatening' in severities:
            return 'life-threatening'
        elif 'severe' in severities:
            return 'severe'
        elif 'moderate' in severities:
            return 'moderate'
        else:
            return 'mild'
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentence_importance(self, sentence: str) -> float:
        """Score sentence importance for summarization"""
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Clinical importance keywords
        clinical_keywords = [
            'adverse', 'event', 'severe', 'serious', 'elevated', 'abnormal',
            'patient', 'treatment', 'medication', 'dose', 'therapy', 'outcome',
            'safety', 'efficacy', 'response', 'toxicity', 'side effect'
        ]
        
        for keyword in clinical_keywords:
            if keyword in sentence_lower:
                score += 1.0
        
        # Lab values and numbers add importance
        if re.search(r'\d+', sentence):
            score += 0.5
        
        # Longer sentences might be more informative
        score += len(sentence.split()) * 0.1
        
        return score
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key clinical points"""
        key_points = []
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Look for key clinical indicators
            if any(keyword in sentence_lower for keyword in ['adverse', 'severe', 'elevated', 'abnormal', 'serious']):
                key_points.append(sentence.strip())
            
            # Look for specific measurements
            if re.search(r'\d+(?:\.\d+)?\s*(mg/dl|u/l|mmhg)', sentence_lower):
                key_points.append(sentence.strip())
        
        return key_points[:5]  # Limit to top 5 key points
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the NLP engine"""
        return {
            "engine_type": "rule_based_clinical_nlp",
            "version": "1.0.0",
            "capabilities": [
                "adverse_event_extraction",
                "lab_abnormality_detection", 
                "protocol_deviation_detection",
                "clinical_text_summarization"
            ],
            "ae_terms_count": len(self.ae_terms),
            "lab_tests_supported": len(self.lab_abnormalities),
            "ready": True
        }
