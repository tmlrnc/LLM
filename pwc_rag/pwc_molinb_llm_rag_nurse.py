"""
Healthcare Multi-Agent RAG System with Embedding Optimization and Re-ranking
===========================================================================

This system implements:
1. Domain-specific embeddings using BioBERT/ClinicalBERT
2. Hierarchical re-ranking framework
3. Cross-domain knowledge integration
4. Hallucination reduction strategies
5. Multi-agent architecture for specialized medical domains
6. Vector database category optimization
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

# For embeddings and NLP
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import spacy

# For vector database
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# For retrieval
from rank_bm25 import BM25Okapi
from langchain.retrievers import BM25Retriever
from langchain.schema import Document

# For LLM integration
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# For medical entity recognition
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

# For FHIR integration
from fhirclient import client
from fhirclient.models import patient, observation, medicationrequest

# Configuration
@dataclass
class HealthcareRAGConfig:
    """Configuration for the Healthcare RAG system"""
    # Model configs
    llm_model_name: str = "gpt-4"
    embedding_model_name: str = "jackon/clinicalbert"
    medical_nlp_model: str = "en_core_sci_lg"
    
    # Database configs
    vector_db_path: str = "./vector_db"
    knowledge_base_path: str = "./knowledge_base"
    
    # Retrieval configs
    initial_retrieval_k: int = 50
    first_pass_k: int = 20
    deep_rerank_k: int = 10
    final_results_k: int = 5
    
    # Agent configs
    agent_specializations: List[str] = None
    confidence_threshold: float = 0.7
    
    def __post_init__(self):
        if self.agent_specializations is None:
            self.agent_specializations = [
                "diabetes_management",
                "cardiology",
                "neurology",
                "general_practice"
            ]


#######################
# Embedding Optimization
#######################

class MedicalEmbeddingModel:
    """Domain-specific embedding model for healthcare data"""
    
    def __init__(self, model_name="jackon/clinicalbert"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_sci_lg")
        
        # Add abbreviation detector and entity linker
        self.nlp.add_pipe("abbreviation_detector")
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    
    def encode(self, texts: List[str], batch_size=8) -> np.ndarray:
        """Encode texts into embeddings"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded_input = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Use CLS token as embedding
                batch_embeddings = model_output.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
                
        return np.vstack(embeddings)
    
    def enrich_with_entities(self, text: str) -> str:
        """Enrich text with recognized medical entities"""
        doc = self.nlp(text)
        entities = []
        
        # Extract entities and their UMLS links
        for ent in doc.ents:
            entities.append(f"{ent.text} [Type: {ent.label_}]")
            
        # Extract abbreviations
        for abrv in doc._.abbreviations:
            entities.append(f"{abrv} -> {abrv._.long_form}")
        
        # Add entity information to text
        enriched_text = text
        if entities:
            enriched_text += "\n\nEntities:\n" + "\n".join(entities)
            
        return enriched_text


#######################
# Hierarchical Re-Ranking Framework
#######################

class HierarchicalRanker:
    """Multi-stage ranking pipeline for medical information retrieval"""
    
    def __init__(self, config: HealthcareRAGConfig):
        self.config = config
        self.embedding_model = MedicalEmbeddingModel(config.embedding_model_name)
        
        # BM25 for initial retrieval
        self.bm25_retriever = None  # Will be initialized when documents are loaded
        
        # Embedding models for different stages
        self.lightweight_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.deep_model = SentenceTransformer("all-mpnet-base-v2")
        
        # Medical domain knowledge rules
        self.domain_rules = self._load_domain_rules()
    
    def _load_domain_rules(self) -> Dict:
        """Load domain-specific rules for re-ranking"""
        # In production, these would be loaded from a database or file
        return {
            "diabetes": {
                "boost_terms": ["glucose", "insulin", "A1C", "metformin", "hyperglycemia"],
                "entity_weights": {"medication": 1.5, "condition": 1.2, "procedure": 1.0}
            },
            "cardiology": {
                "boost_terms": ["heart", "cardiac", "ECG", "blood pressure", "cholesterol"],
                "entity_weights": {"medication": 1.2, "condition": 1.5, "procedure": 1.3}
            }
        }
    
    def initialize_with_corpus(self, documents: List[Dict[str, Any]]):
        """Initialize retrievers with document corpus"""
        # Prepare corpus for BM25
        tokenized_corpus = [doc["content"].lower().split() for doc in documents]
        self.bm25_retriever = BM25Okapi(tokenized_corpus)
        
        # Store original documents
        self.documents = documents
    
    def retrieve_and_rank(self, query: str, domain: str = None) -> List[Dict[str, Any]]:
        """Full retrieval and ranking pipeline"""
        # Step 1: Initial broad retrieval with BM25
        tokenized_query = query.lower().split()
        initial_scores = self.bm25_retriever.get_scores(tokenized_query)
        initial_indices = np.argsort(initial_scores)[::-1][:self.config.initial_retrieval_k]
        initial_results = [self.documents[i] for i in initial_indices]
        
        # Step 2: First-pass ranking with lightweight embedding model
        query_embedding = self.lightweight_model.encode(query, show_progress_bar=False)
        first_pass_embeddings = self.lightweight_model.encode(
            [doc["content"] for doc in initial_results],
            show_progress_bar=False
        )
        first_pass_scores = np.dot(first_pass_embeddings, query_embedding)
        first_pass_indices = np.argsort(first_pass_scores)[::-1][:self.config.first_pass_k]
        first_pass_results = [initial_results[i] for i in first_pass_indices]
        
        # Step 3: Deep re-ranking with cross-attention models
        query_embedding_deep = self.deep_model.encode(query, show_progress_bar=False)
        deep_rerank_embeddings = self.deep_model.encode(
            [doc["content"] for doc in first_pass_results],
            show_progress_bar=False
        )
        deep_rerank_scores = np.dot(deep_rerank_embeddings, query_embedding_deep)
        deep_rerank_indices = np.argsort(deep_rerank_scores)[::-1][:self.config.deep_rerank_k]
        deep_rerank_results = [first_pass_results[i] for i in deep_rerank_indices]
        
        # Step 4: Final re-ranking with domain-specific rules
        final_results = self._apply_domain_rules(deep_rerank_results, query, domain)
        
        return final_results[:self.config.final_results_k]
    
    def _apply_domain_rules(self, candidates: List[Dict[str, Any]], query: str, domain: str = None) -> List[Dict[str, Any]]:
        """Apply domain-specific rules and heuristics"""
        if domain is None or domain not in self.domain_rules:
            return candidates
        
        rules = self.domain_rules[domain]
        
        # Initialize NLP for entity extraction
        nlp = self.embedding_model.nlp
        
        # Process documents
        for doc in candidates:
            score_modifier = 1.0
            
            # Check for boost terms
            for term in rules["boost_terms"]:
                if term.lower() in doc["content"].lower():
                    score_modifier += 0.1
            
            # Apply entity weights
            doc_nlp = nlp(doc["content"])
            for ent in doc_nlp.ents:
                entity_type = ent.label_.lower()
                if entity_type in rules["entity_weights"]:
                    score_modifier += 0.05 * rules["entity_weights"][entity_type]
            
            # Apply score modifier
            doc["final_score"] = doc.get("score", 1.0) * score_modifier
        
        # Sort by final score
        candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return candidates


#######################
# Hallucination Reduction Strategies
#######################

class FactualityVerifier:
    """Verify factuality of generated content against trusted sources"""
    
    def __init__(self, config: HealthcareRAGConfig):
        self.config = config
        self.knowledge_base = self._load_knowledge_base()
        self.llm = ChatOpenAI(model_name=config.llm_model_name)
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load clinician-validated knowledge base"""
        knowledge_base = {}
        kb_path = self.config.knowledge_base_path
        
        if os.path.exists(kb_path):
            for filename in os.listdir(kb_path):
                if filename.endswith('.json'):
                    with open(os.path.join(kb_path, filename), 'r') as f:
                        domain_knowledge = json.load(f)
                        knowledge_base[filename.replace('.json', '')] = domain_knowledge
        
        return knowledge_base
    
    def verify_content(self, generated_text: str, domain: str = None) -> Tuple[str, Dict[str, Any]]:
        """Verify factuality of generated content"""
        # Extract claims from generated text
        claims = self._extract_claims(generated_text)
        
        # Verify each claim
        verification_results = {}
        for claim in claims:
            verification = self._verify_claim(claim, domain)
            verification_results[claim] = verification
        
        # Annotate generated text with confidence scores
        annotated_text = self._annotate_text(generated_text, verification_results)
        
        # Create verification metadata
        metadata = {
            "verification_results": verification_results,
            "overall_confidence": self._calculate_overall_confidence(verification_results),
            "trusted_sources": self._get_cited_sources(verification_results)
        }
        
        return annotated_text, metadata
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Use LLM to extract claims
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract the key medical factual claims from the following text. 
            For each claim, provide it as a single sentence statement of fact.
            
            Text: {text}
            
            Key factual claims:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(text=text)
        
        # Process result
        claims = [claim.strip() for claim in result.split("\n") if claim.strip()]
        return claims
    
    def _verify_claim(self, claim: str, domain: str = None) -> Dict[str, Any]:
        """Verify a single claim against knowledge base and trusted sources"""
        # Check knowledge base first
        kb_result = self._check_knowledge_base(claim, domain)
        
        if kb_result["confidence"] > self.config.confidence_threshold:
            return kb_result
        
        # If not found or low confidence, use LLM to verify
        llm_result = self._verify_with_llm(claim)
        
        # Combine results
        final_confidence = max(kb_result["confidence"], llm_result["confidence"])
        final_sources = kb_result["sources"] + llm_result["sources"]
        
        return {
            "confidence": final_confidence,
            "sources": final_sources,
            "verified": final_confidence > self.config.confidence_threshold
        }
    
    def _check_knowledge_base(self, claim: str, domain: str = None) -> Dict[str, Any]:
        """Check claim against structured knowledge base"""
        if domain and domain in self.knowledge_base:
            domain_kb = self.knowledge_base[domain]
            
            # Simple exact matching for demonstration
            # In production, this would use semantic matching
            for entry in domain_kb.get("facts", []):
                if entry["statement"].lower() == claim.lower():
                    return {
                        "confidence": 1.0,
                        "sources": entry.get("sources", []),
                        "verified": True
                    }
        
        # Check general knowledge base
        if "general" in self.knowledge_base:
            for entry in self.knowledge_base["general"].get("facts", []):
                if entry["statement"].lower() == claim.lower():
                    return {
                        "confidence": 0.9,
                        "sources": entry.get("sources", []),
                        "verified": True
                    }
        
        # Not found
        return {
            "confidence": 0.0,
            "sources": [],
            "verified": False
        }
    
    def _verify_with_llm(self, claim: str) -> Dict[str, Any]:
        """Use LLM to verify claim and provide sources"""
        prompt = PromptTemplate(
            input_variables=["claim"],
            template="""
            Verify the following medical claim against your knowledge of established medical facts:
            
            Claim: {claim}
            
            Please rate the confidence of this claim being factually correct from 0.0 to 1.0,
            where 0.0 means definitely incorrect and 1.0 means definitely correct.
            
            Also provide any specific authoritative medical sources that would support or refute this claim.
            
            Format your response as JSON:
            {{
                "confidence": <float>,
                "explanation": "<explanation>",
                "sources": ["<source1>", "<source2>", ...]
            }}
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(claim=claim)
        
        try:
            # Extract JSON
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            json_str = result[json_start:json_end]
            verification = json.loads(json_str)
            
            return verification
        except:
            # Fallback if JSON parsing fails
            return {
                "confidence": 0.5,
                "sources": [],
                "verified": False,
                "explanation": "Could not verify claim due to processing error."
            }
    
    def _annotate_text(self, text: str, verification_results: Dict[str, Any]) -> str:
        """Annotate text with confidence scores"""
        annotated_text = text
        
        # Add footnotes for each claim
        footnotes = []
        for i, (claim, verification) in enumerate(verification_results.items(), 1):
            confidence = verification["confidence"]
            confidence_label = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
            
            # Find the claim in text and add a footnote marker
            if claim in annotated_text:
                annotated_text = annotated_text.replace(claim, f"{claim}[{i}]", 1)
            
            # Create footnote
            footnote = f"[{i}] Confidence: {confidence_label} ({confidence:.2f})"
            if verification.get("sources"):
                sources = ", ".join(verification["sources"][:2])  # Limit to first 2 sources
                footnote += f" | Sources: {sources}"
            
            footnotes.append(footnote)
        
        # Add footnotes at the end
        if footnotes:
            annotated_text += "\n\n---\n" + "\n".join(footnotes)
        
        return annotated_text
    
    def _calculate_overall_confidence(self, verification_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the generated content"""
        if not verification_results:
            return 0.5
        
        confidences = [v["confidence"] for v in verification_results.values()]
        return sum(confidences) / len(confidences)
    
    def _get_cited_sources(self, verification_results: Dict[str, Any]) -> List[str]:
        """Get all cited sources from verification results"""
        all_sources = []
        for verification in verification_results.values():
            all_sources.extend(verification.get("sources", []))
        
        # Remove duplicates
        return list(set(all_sources))


#######################
# Multi-Agent Architecture
#######################

class MedicalAgent:
    """Base class for specialized medical agents"""
    
    def __init__(self, config: HealthcareRAGConfig, domain: str):
        self.config = config
        self.domain = domain
        self.llm = ChatOpenAI(model_name=config.llm_model_name)
        self.ranker = None  # Will be set by manager
        self.verifier = None  # Will be set by manager
    
    def set_services(self, ranker: HierarchicalRanker, verifier: FactualityVerifier):
        """Set shared services"""
        self.ranker = ranker
        self.verifier = verifier
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query and generate a response"""
        # Retrieve relevant information
        retrieved_docs = self.ranker.retrieve_and_rank(query, self.domain)
        
        # Generate response
        response = self._generate_response(query, retrieved_docs, context)
        
        # Verify response
        verified_response, verification_metadata = self.verifier.verify_content(response, self.domain)
        
        return {
            "domain": self.domain,
            "query": query,
            "retrieved_docs": retrieved_docs,
            "response": verified_response,
            "verification": verification_metadata,
            "confidence": verification_metadata["overall_confidence"]
        }
    
    def _generate_response(self, query: str, docs: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """Generate response using LLM"""
        context_str = self._format_context(context) if context else ""
        docs_str = "\n\n".join([f"Document {i+1}: {doc['content']}" for i, doc in enumerate(docs)])
        
        prompt = PromptTemplate(
            input_variables=["query", "context", "docs", "domain"],
            template="""
            You are a specialized healthcare AI assistant focusing on {domain}. 
            
            Patient Context:
            {context}
            
            User Query:
            {query}
            
            Relevant Medical Information:
            {docs}
            
            Please provide a comprehensive, factual response to the query based on the provided information.
            Focus on evidence-based recommendations and include appropriate medical context.
            Be clear about confidence levels and, when appropriate, suggest when a healthcare provider should be consulted.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            query=query,
            context=context_str,
            docs=docs_str,
            domain=self.domain
        )
        
        return response
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format patient context information"""
        if not context:
            return ""
        
        context_str = "Patient Information:\n"
        
        if "demographics" in context:
            demographics = context["demographics"]
            context_str += f"Age: {demographics.get('age', 'Unknown')}\n"
            context_str += f"Sex: {demographics.get('sex', 'Unknown')}\n"
        
        if "conditions" in context:
            context_str += "\nMedical Conditions:\n"
            for condition in context["conditions"]:
                context_str += f"- {condition}\n"
        
        if "medications" in context:
            context_str += "\nCurrent Medications:\n"
            for medication in context["medications"]:
                context_str += f"- {medication}\n"
        
        return context_str


class DiabetesManagementAgent(MedicalAgent):
    """Specialized agent for diabetes management"""
    
    def __init__(self, config: HealthcareRAGConfig):
        super().__init__(config, "diabetes_management")
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process diabetes-specific query"""
        # Enrich context with diabetes-specific information
        enriched_context = self._enrich_diabetes_context(context) if context else None
        
        # Use base class to process
        result = super().process_query(query, enriched_context)
        
        # Add diabetes-specific insights
        if "blood_glucose" in enriched_context:
            glucose_insights = self._analyze_glucose_patterns(enriched_context["blood_glucose"])
            result["glucose_insights"] = glucose_insights
        
        return result
    
    def _enrich_diabetes_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich context with diabetes-specific information"""
        enriched = context.copy()
        
        # Add diabetes type if not specified
        if "conditions" in enriched:
            has_diabetes = False
            for condition in enriched["conditions"]:
                if "diabetes" in condition.lower():
                    has_diabetes = True
                    break
            
            if not has_diabetes:
                enriched["conditions"].append("Unspecified diabetes")
        
        return enriched
    
    def _analyze_glucose_patterns(self, glucose_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze blood glucose patterns"""
        if not glucose_data:
            return {"message": "No glucose data available for analysis"}
        
        # Convert to pandas for analysis
        df = pd.DataFrame(glucose_data)
        
        # Basic statistics
        stats = {
            "average": df["value"].mean(),
            "min": df["value"].min(),
            "max": df["value"].max(),
            "in_range_percent": (df["value"].between(70, 180).sum() / len(df)) * 100
        }
        
        # Time-based patterns
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["datetime"].dt.hour
            
            # Morning highs detection
            morning_readings = df[df["hour"].between(4, 9)]
            if len(morning_readings) > 0:
                stats["morning_average"] = morning_readings["value"].mean()
                stats["dawn_effect"] = stats["morning_average"] > stats["average"]
        
        return stats


#######################
# Multi-Agent Manager
#######################

class HealthcareMultiAgentSystem:
    """Manager for the multi-agent healthcare system"""
    
    def __init__(self, config: HealthcareRAGConfig = None):
        self.config = config or HealthcareRAGConfig()
        
        # Initialize vector database
        self.vector_db = self._setup_vector_db()
        
        # Initialize shared services
        self.ranker = HierarchicalRanker(self.config)
        self.verifier = FactualityVerifier(self.config)
        
        # Initialize specialized agents
        self.agents = self._setup_agents()
    
    def _setup_vector_db(self):
        """Setup vector database with optimized categories"""
        # In a real implementation, this would initialize a proper vector DB
        # For this example, we'll simulate it
        return {
            "symptom_vectors": {},
            "treatment_vectors": {},
            "outcome_vectors": {},
            "temporal_vectors": {},
        }
    
    def _setup_agents(self) -> Dict[str, MedicalAgent]:
        """Setup specialized medical agents"""
        agents = {}
        
        # Create agents based on specializations in config
        for specialization in self.config.agent_specializations:
            if specialization == "diabetes_management":
                agent = DiabetesManagementAgent(self.config)
            else:
                # Generic agent for other specializations
                agent = MedicalAgent(self.config, specialization)
            
            # Set shared services
            agent.set_services(self.ranker, self.verifier)
            
            agents[specialization] = agent
        
        return agents
    
    def load_document_corpus(self, documents: List[Dict[str, Any]]):
        """Load document corpus into ranking system"""
        self.ranker.initialize_with_corpus(documents)
    
    def process_query(self, query: str, patient_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a healthcare query through the multi-agent system"""
        # Determine which agent(s) should handle the query
        responsible_agents = self._route_query(query, patient_context)
        
        # Collect responses from each relevant agent
        agent_responses = {}
        for agent_name in responsible_agents:
            agent = self.agents.get(agent_name)
            if agent:
                agent_responses[agent_name] = agent.process_query(query, patient_context)
        
        # Combine responses if multiple agents
        if len(agent_responses) > 1:
            combined_response = self._combine_responses(agent_responses, query)
            return combined_response
        elif len(agent_responses) == 1:
            # Return the single agent response
            return next(iter(agent_responses.values()))
        else:
            # Fallback to general practice if no specific agent was selected
            general_agent = self.agents.get("general_practice")
            if general_agent:
                return general_agent.process_query(query, patient_context)
            
            # Last resort fallback
            return {
                "query": query,
                "response": "I'm unable to process this healthcare query at the moment. Please consult with a healthcare professional.",
                "confidence": 0.0
            }
    
    def _route_query(self, query: str, patient_context: Dict[str, Any] = None) -> List[str]:
        """Route query to appropriate specialized agent(s)"""
        # In production, this would use a more sophisticated routing mechanism
        # For this example, we'll use a simple keyword-based approach
        
        diabetes_keywords = ["diabetes", "glucose", "insulin", "a1c", "metformin", "blood sugar"]
        cardiology_keywords = ["heart", "cardiac", "chest pain", "blood pressure", "cholesterol"]
        neurology_keywords = ["brain", "neurological", "headache", "seizure", "stroke"]
        
        # Check query for keywords
        query_lower = query.lower()
        matched_agents = []
        
        if any(keyword in query_lower for keyword in diabetes_keywords):
            matched_agents.append("diabetes_management")
        
        if any(keyword in query_lower for keyword in cardiology_keywords):
            matched_agents.append("cardiology")
        
        if any(keyword in query_lower for keyword in neurology_keywords):
            matched_agents.append("neurology")
        
        # Check patient context if no matches in query
        if not matched_agents and patient_context:
            conditions = patient_context.get("conditions", [])
            for condition in conditions:
                condition_lower = condition.lower()
                
                if any(keyword in condition_lower for keyword in diabetes_keywords):
                    matched_agents.append("diabetes_management")
                
                if any(keyword in condition_lower for keyword in cardiology_keywords):
                    matched_agents.append("cardiology")
                
                if any(keyword in condition_lower for keyword in neurology_keywords):
                    matched_agents.append("neurology")
        
        # Default to general practice if no specific matches
        if not matched_agents:
            matched_agents.append("general_practice")
        
        return matched_agents
    
    def _combine_responses(self, agent_responses: Dict[str, Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Combine responses from multiple agents"""
        # Extract individual responses
        responses = {agent: response["response"] for agent, response in agent_responses.items()}
        confidences = {agent: response["confidence"] for agent, response in agent_responses.items()}
        
        # Find the agent with highest confidence
        primary_agent = max(confidences, key=confidences.get)
        
        # Use LLM to combine responses
        combined_text = self._merge_responses_with_llm(query, responses)
        
        # Verify the combined response
        verified_response, verification_metadata = self.verifier.verify_content(combined_text)
        
        return {
            "query": query,
            "response": verified_response,
            "verification": verification_metadata,
            "confidence": verification_metadata["overall_confidence"],
            "contributing_agents": list(agent_responses.keys()),
            "primary_agent": primary_agent
        }
    
    def _merge_responses_with_llm(self, query: str, responses: Dict[str, str]) -> str:
        """Use LLM to merge multiple agent responses"""
        responses_str = "\n\n".join([f"{agent} response:\n{response}" for agent, response in responses.items()])
        
        prompt = PromptTemplate(
            input_variables=["query", "responses"],
            template="""
            You are a specialized healthcare AI assistant. You have received multiple expert responses to the following query:
            
            Query: {query}
            
            Expert Responses:
            {responses}
            
            Please synthesize these responses into a single, comprehensive answer that:
            1. Incorporates all relevant medical information
            2. Resolves any contradictions between expert responses
            3. Maintains clinical accuracy and context
            4. Prioritizes patient safety and evidence-based guidance
            
            Your synthesized response:
            """
        )
        
        llm = ChatOpenAI(model_name=self.config.llm_model_name)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        combined_response = chain.run(
            query=query,
            responses=responses_str
        )
        
        return combined_response


#######################
# FHIR Integration for Clinical Workflows
#######################

class FHIRIntegration:
    """Integration with FHIR for clinical workflows"""
    
    def __init__(self, fhir_server_url: str, patient_id: str = None):
        self.fhir_client = client.FHIRClient(settings={
            'app_id': 'healthcare_rag_system',
            'api_base': fhir_server_url
        })
        self.patient_id = patient_id
    
    def set_patient(self, patient_id: str):
        """Set the current patient ID"""
        self.patient_id = patient_id
    
    def get_patient_context(self) -> Dict[str, Any]:
        """Retrieve patient context from FHIR server"""
        if not self.patient_id:
            return {}
        
        context = {
            "demographics": self._get_demographics(),
            "conditions": self._get_conditions(),
            "medications": self._get_medications(),
            "observations": self._get_observations()
        }
        
        return context
    
    def _get_demographics(self) -> Dict[str, Any]:
        """Get patient demographics"""
        try:
            patient_data = patient.Patient.read(self.patient_id, self.fhir_client.server)
            
            demographics = {
                "id": patient_data.id,
                "name": f"{patient_data.name[0].given[0]} {patient_data.name[0].family}",
                "gender": patient_data.gender,
                "birthDate": patient_data.birthDate.isostring
            }
            
            # Calculate age
            from datetime import datetime
            birth_date = datetime.fromisoformat(patient_data.birthDate.isostring)
            current_date = datetime.now()
            age = current_date.year - birth_date.year
            
            if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
                age -= 1
                
            demographics["age"] = age
            
            return demographics
        except Exception as e:
            print(f"Error retrieving demographics: {e}")
            return {}
    
    def _get_conditions(self) -> List[str]:
        """Get patient conditions"""
        try:
            search = self.fhir_client.server.search(
                'Condition',
                params={
                    'patient': self.patient_id,
                    'clinical-status': 'active'
                }
            )
            
            conditions = []
            for condition in search.entry:
                condition_resource = condition.resource
                conditions.append(condition_resource.code.coding[0].display)
            
            return conditions
        except Exception as e:
            print(f"Error retrieving conditions: {e}")
            return []
    
    def _get_medications(self) -> List[str]:
        """Get patient medications"""
        try:
            search = self.fhir_client.server.search(
                'MedicationRequest',
                params={
                    'patient': self.patient_id,
                    'status': 'active'
                }
            )
            
            medications = []
            for med_request in search.entry:
                med_resource = med_request.resource
                medication_name = med_resource.medicationCodeableConcept.coding[0].display
                dosage = med_resource.dosageInstruction[0].text
                medications.append(f"{medication_name} - {dosage}")
            
            return medications
        except Exception as e:
            print(f"Error retrieving medications: {e}")
            return []
    
    def _get_observations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get patient observations"""
        try:
            search = self.fhir_client.server.search(
                'Observation',
                params={
                    'patient': self.patient_id,
                    '_count': '100',
                    '_sort': '-date'
                }
            )
            
            observations = {}
            
            for obs_entry in search.entry:
                obs = obs_entry.resource
                
                # Skip if missing key information
                if not hasattr(obs, 'code') or not hasattr(obs, 'valueQuantity'):
                    continue
                
                code = obs.code.coding[0].code
                display = obs.code.coding[0].display
                value = obs.valueQuantity.value
                unit = obs.valueQuantity.unit
                date = obs.effectiveDateTime.isostring
                
                if code not in observations:
                    observations[code] = []
                
                observations[code].append({
                    "name": display,
                    "value": value,
                    "unit": unit,
                    "timestamp": date
                })
            
            # Group observations by type
            result = {}
            
            # Blood glucose readings
            glucose_codes = ["2339-0", "2345-7", "41653-7"]  # Various glucose measurement codes
            glucose_readings = []
            
            for code in glucose_codes:
                if code in observations:
                    glucose_readings.extend(observations[code])
            
            if glucose_readings:
                result["blood_glucose"] = glucose_readings
            
            # Blood pressure readings
            bp_codes = ["85354-9", "8480-6", "8462-4"]  # BP panel, systolic, diastolic
            bp_readings = []
            
            for code in bp_codes:
                if code in observations:
                    bp_readings.extend(observations[code])
            
            if bp_readings:
                result["blood_pressure"] = bp_readings
            
            return result
        except Exception as e:
            print(f"Error retrieving observations: {e}")
            return {}
    
    def create_notification(self, message: str, urgency: str = "routine") -> bool:
        """Create a notification for the healthcare provider"""
        if not self.patient_id:
            return False
        
        try:
            # Create a communication resource
            from fhirclient.models import communication
            
            comm = communication.Communication({
                "status": "in-progress",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/communication-category",
                        "code": "alert",
                        "display": "Alert"
                    }]
                }],
                "priority": urgency,
                "subject": {
                    "reference": f"Patient/{self.patient_id}"
                },
                "sent": datetime.now().isoformat(),
                "payload": [{
                    "contentString": message
                }]
            })
            
            # Save the communication
            result = comm.create(self.fhir_client.server)
            return result is not None
        
        except Exception as e:
            print(f"Error creating notification: {e}")
            return False


#######################
# Main Implementation
#######################

def create_healthcare_rag_system(config_path: str = None) -> HealthcareMultiAgentSystem:
    """Create and initialize the healthcare RAG system"""
    # Load configuration
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            config = HealthcareRAGConfig(**config_dict)
    else:
        config = HealthcareRAGConfig()
    
    # Create the system
    system = HealthcareMultiAgentSystem(config)
    
    # Load sample documents if available
    sample_docs = load_sample_documents()
    if sample_docs:
        system.load_document_corpus(sample_docs)
    
    return system


def load_sample_documents() -> List[Dict[str, Any]]:
    """Load sample medical documents for demonstration"""
    # In production, these would be loaded from a database
    sample_docs = [
        {
            "id": "diabetes-001",
            "title": "Type 2 Diabetes Management Guidelines",
            "content": "Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency. Management focuses on lifestyle modifications, medication adherence, and regular blood glucose monitoring. Target A1C is typically <7% for most patients, though this may be individualized. First-line pharmacotherapy is usually metformin, with additional agents added as needed. Regular monitoring for complications including nephropathy, retinopathy, and neuropathy is essential."
        },
        {
            "id": "diabetes-002",
            "title": "Blood Glucose Monitoring Best Practices",
            "content": "Blood glucose monitoring is essential for diabetes management. Recommended testing frequency varies by treatment regimen, with insulin-dependent patients typically testing 2-4 times daily. Target glucose ranges are 80-130 mg/dL preprandial and <180 mg/dL postprandial. CGM systems provide real-time glucose data and can improve time in range. Proper technique includes washing hands, using sufficient blood samples, and calibrating meters regularly."
        },
        {
            "id": "cardio-001",
            "title": "Hypertension Management in Primary Care",
            "content": "Hypertension is defined as blood pressure ≥130/80 mmHg. Initial evaluation should include risk assessment for cardiovascular disease. Lifestyle modifications include DASH diet, sodium restriction, physical activity, and weight management. Pharmacotherapy typically begins with thiazide diuretics, ACE inhibitors, ARBs, or calcium channel blockers. Target BP is <130/80 mmHg for most patients. Regular monitoring and medication adjustment is essential for optimal control."
        }
    ]
    
    return sample_docs


def example_usage():
    """Example usage of the healthcare RAG system"""
    # Create the system
    system = create_healthcare_rag_system()
    
    # Sample patient context
    patient_context = {
        "demographics": {
            "age": 58,
            "gender": "female"
        },
        "conditions": [
            "Type 2 Diabetes Mellitus",
            "Hypertension",
            "Hyperlipidemia"
        ],
        "medications": [
            "Metformin 1000mg twice daily",
            "Lisinopril 10mg daily",
            "Atorvastatin 20mg daily"
        ],
        "blood_glucose": [
            {"value": 142, "unit": "mg/dL", "timestamp": "2023-03-15T08:00:00"},
            {"value": 165, "unit": "mg/dL", "timestamp": "2023-03-15T12:00:00"},
            {"value": 138, "unit": "mg/dL", "timestamp": "2023-03-15T18:00:00"}
        ]
    }
    
    # Sample query
    query = "What should I do if my blood sugar readings are consistently above 180?"
    
    # Process the query
    result = system.process_query(query, patient_context)
    
    # Print the result
    print("Query:", query)
    print("\nResponse:", result["response"])
    print("\nConfidence:", result["confidence"])
    
    # If using multiple agents
    if "contributing_agents" in result:
        print("\nContributing Agents:", ", ".join(result["contributing_agents"]))
    
    # If glucose insights are available
    if "glucose_insights" in result:
        print("\nGlucose Insights:", result["glucose_insights"])


if __name__ == "__main__":
    example_usage()
