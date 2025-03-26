"""
Oncology Decision Support System: FHIR-based LLM RAG Architecture
POC Implementation
"""

import json
import asyncio
import datetime
import requests
from typing import Dict, List, Optional, Any, Union

import aiohttp
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from sentence_transformers import CrossEncoder

# Configuration
CONFIG = {
    "fhir_server_url": "https://your-fhir-server/fhir",
    "nccn_api_url": "https://api.nccn.org/guidelines/v1/",
    "nccn_api_key": "your-nccn-api-key",
    "openai_api_key": "your-openai-api-key",
    "reranker_model_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "vector_db_path": "./vector_db",
    "fhir_subscription_endpoint": "wss://your-fhir-server/fhir/subscription",
    "kafka_broker": "your-kafka-broker:9092",
    "metadata_weights": {
        "evidence_A1": 0.8,
        "evidence_A2": 0.6,
        "evidence_B": 0.4,
        "evidence_C": 0.2,
        "recency": 0.5,
        "recommended_trial": 0.7,
        "genomic_match": 0.9
    }
}

# Constants
ONCOLOGY_RELEVANT_LOINC_CODES = [
    "21893-3",  # Cancer diagnosis
    "59847-4",  # Stage group.pathology
    "21905-5",  # Tumor size
    "21906-3",  # Tumor grade
    "21918-8"   # Genetic markers
]

CHEMOTHERAPY_RXNORM_CODES = [
    "1156676",  # Gemcitabine
    "1156827",  # Cisplatin
    "1156843",  # Paclitaxel
    "1156214"   # Rituximab
]

# Classes and Functions
class NCCNGuidelinesAPI:
    """Access NCCN guidelines through API or structured data"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def get_guideline(self, cancer_type: str, version: str = "latest") -> Dict:
        """Retrieve guidelines for a specific cancer type"""
        endpoint = f"guidelines/{cancer_type}"
        params = {"version": version}
        response = self.session.get(f"{self.base_url}{endpoint}", params=params)
        return response.json()
    
    def search_recommendations(self, query: str) -> Dict:
        """Search for specific treatment recommendations"""
        endpoint = "search"
        params = {"q": query}
        response = self.session.get(f"{self.base_url}{endpoint}", params=params)
        return response.json()
    
    def get_all_cancer_types(self) -> List[str]:
        """Get a list of all available cancer types"""
        endpoint = "cancer-types"
        response = self.session.get(f"{self.base_url}{endpoint}")
        return response.json()["cancer_types"]


class FHIRIntegrationManager:
    """Manages integration with FHIR servers and EMR systems"""
    
    def __init__(self, fhir_base_url: str):
        self.fhir_base_url = fhir_base_url
        self.session = requests.Session()
    
    def get_patient(self, patient_id: str) -> Dict:
        """Get patient resource by ID"""
        endpoint = f"Patient/{patient_id}"
        response = self.session.get(f"{self.fhir_base_url}/{endpoint}")
        return response.json()
    
    def get_conditions(self, patient_id: str, code: Optional[str] = None) -> List[Dict]:
        """Get patient conditions, optionally filtered by code"""
        endpoint = "Condition"
        params = {"patient": patient_id}
        if code:
            params["code"] = code
        response = self.session.get(f"{self.fhir_base_url}/{endpoint}", params=params)
        return response.json().get("entry", [])
    
    def get_observations(self, patient_id: str, category: Optional[str] = None) -> List[Dict]:
        """Get patient observations, optionally filtered by category"""
        endpoint = "Observation"
        params = {"patient": patient_id}
        if category:
            params["category"] = category
        response = self.session.get(f"{self.fhir_base_url}/{endpoint}", params=params)
        return response.json().get("entry", [])
    
    def get_medications(self, patient_id: str) -> List[Dict]:
        """Get patient medication administrations"""
        endpoint = "MedicationAdministration"
        params = {"patient": patient_id}
        response = self.session.get(f"{self.fhir_base_url}/{endpoint}", params=params)
        return response.json().get("entry", [])
    
    def get_patient_oncology_profile(self, patient_id: str) -> Dict:
        """Get comprehensive oncology profile for a patient"""
        patient = self.get_patient(patient_id)
        conditions = self.get_conditions(patient_id)
        observations = self.get_observations(patient_id, "laboratory")
        medications = self.get_medications(patient_id)
        
        # Extract key oncology information
        cancer_conditions = [c for c in conditions if self._is_cancer_condition(c)]
        biomarkers = [o for o in observations if self._is_biomarker(o)]
        cancer_treatments = [m for m in medications if self._is_cancer_treatment(m)]
        
        return {
            "patient": self._format_patient(patient),
            "cancer_diagnosis": self._format_cancer_diagnosis(cancer_conditions),
            "biomarkers": self._format_biomarkers(biomarkers),
            "treatment_history": self._format_treatments(cancer_treatments)
        }
    
    def _is_cancer_condition(self, condition: Dict) -> bool:
        """Check if a condition is cancer-related"""
        # Implementation would check SNOMED or ICD-10 codes
        # This is a simplified placeholder
        if "resource" in condition:
            condition = condition["resource"]
        
        coding = condition.get("code", {}).get("coding", [])
        for code in coding:
            if code.get("system") == "http://snomed.info/sct":
                # Check if code is in oncology range
                if code.get("code", "").startswith("363"):
                    return True
        return False
    
    def _is_biomarker(self, observation: Dict) -> bool:
        """Check if an observation is a relevant biomarker"""
        if "resource" in observation:
            observation = observation["resource"]
        
        loinc_code = self._extract_loinc_code(observation)
        return loinc_code in ONCOLOGY_RELEVANT_LOINC_CODES
    
    def _is_cancer_treatment(self, medication: Dict) -> bool:
        """Check if a medication is cancer treatment"""
        if "resource" in medication:
            medication = medication["resource"]
        
        rxnorm_code = self._extract_rxnorm_code(medication)
        return rxnorm_code in CHEMOTHERAPY_RXNORM_CODES
    
    def _extract_loinc_code(self, observation: Dict) -> Optional[str]:
        """Extract LOINC code from observation"""
        if "resource" in observation:
            observation = observation["resource"]
        
        coding = observation.get("code", {}).get("coding", [])
        for code in coding:
            if code.get("system") == "http://loinc.org":
                return code.get("code")
        return None
    
    def _extract_rxnorm_code(self, medication: Dict) -> Optional[str]:
        """Extract RxNorm code from medication"""
        if "resource" in medication:
            medication = medication["resource"]
        
        if "medicationCodeableConcept" in medication:
            coding = medication.get("medicationCodeableConcept", {}).get("coding", [])
            for code in coding:
                if code.get("system") == "http://www.nlm.nih.gov/research/umls/rxnorm":
                    return code.get("code")
        return None
    
    def _format_patient(self, patient: Dict) -> Dict:
        """Format patient data for the oncology profile"""
        if "resource" in patient:
            patient = patient["resource"]
        
        return {
            "id": patient.get("id"),
            "gender": patient.get("gender"),
            "birthDate": patient.get("birthDate"),
            "age": self._calculate_age(patient.get("birthDate"))
        }
    
    def _format_cancer_diagnosis(self, conditions: List[Dict]) -> List[Dict]:
        """Format cancer diagnoses for the oncology profile"""
        formatted = []
        for condition in conditions:
            if "resource" in condition:
                condition = condition["resource"]
            
            formatted.append({
                "code": self._get_code_display(condition.get("code", {})),
                "onsetDate": condition.get("onsetDateTime"),
                "stage": self._extract_stage(condition),
                "status": condition.get("clinicalStatus", {}).get("coding", [{}])[0].get("code")
            })
        return formatted
    
    def _format_biomarkers(self, observations: List[Dict]) -> List[Dict]:
        """Format biomarkers for the oncology profile"""
        formatted = []
        for observation in observations:
            if "resource" in observation:
                observation = observation["resource"]
            
            formatted.append({
                "code": self._get_code_display(observation.get("code", {})),
                "value": self._extract_observation_value(observation),
                "date": observation.get("effectiveDateTime"),
                "interpretation": self._extract_interpretation(observation)
            })
        return formatted
    
    def _format_treatments(self, medications: List[Dict]) -> List[Dict]:
        """Format treatments for the oncology profile"""
        formatted = []
        for medication in medications:
            if "resource" in medication:
                medication = medication["resource"]
            
            med_code = medication.get("medicationCodeableConcept", {})
            formatted.append({
                "medication": self._get_code_display(med_code),
                "date": medication.get("effectiveDateTime"),
                "dosage": self._extract_dosage(medication),
                "route": self._extract_route(medication)
            })
        return formatted
    
    def _get_code_display(self, code_obj: Dict) -> str:
        """Get display text from a codeable concept"""
        if code_obj.get("text"):
            return code_obj.get("text")
        
        coding = code_obj.get("coding", [])
        if coding and coding[0].get("display"):
            return coding[0].get("display")
        
        return "Unknown"
    
    def _extract_observation_value(self, observation: Dict) -> Union[str, float, None]:
        """Extract value from observation"""
        if "valueQuantity" in observation:
            value = observation["valueQuantity"].get("value")
            unit = observation["valueQuantity"].get("unit", "")
            return f"{value} {unit}".strip()
        elif "valueString" in observation:
            return observation["valueString"]
        elif "valueCodeableConcept" in observation:
            return self._get_code_display(observation["valueCodeableConcept"])
        return None
    
    def _extract_interpretation(self, observation: Dict) -> Optional[str]:
        """Extract interpretation from observation"""
        if "interpretation" in observation:
            return self._get_code_display(observation["interpretation"][0])
        return None
    
    def _extract_stage(self, condition: Dict) -> Optional[str]:
        """Extract cancer stage from condition"""
        # In real implementation, would look for stage in FHIR extension
        # This is a simplified placeholder
        return None
    
    def _extract_dosage(self, medication: Dict) -> Optional[str]:
        """Extract dosage from medication administration"""
        if "dosage" in medication:
            quantity = medication["dosage"].get("dose", {})
            value = quantity.get("value", "")
            unit = quantity.get("unit", "")
            return f"{value} {unit}".strip()
        return None
    
    def _extract_route(self, medication: Dict) -> Optional[str]:
        """Extract administration route from medication administration"""
        if "dosage" in medication and "route" in medication["dosage"]:
            return self._get_code_display(medication["dosage"]["route"])
        return None
    
    def _calculate_age(self, birth_date: Optional[str]) -> Optional[int]:
        """Calculate age from birth date"""
        if not birth_date:
            return None
        
        try:
            birth_date = datetime.datetime.strptime(birth_date, "%Y-%m-%d")
            today = datetime.datetime.now()
            age = today.year - birth_date.year
            if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
                age -= 1
            return age
        except ValueError:
            return None


class VectorDBManager:
    """Manages vector database operations for guideline storage and retrieval"""
    
    def __init__(self, db_path: str, embedding_model_name: str = "text-embedding-ada-002"):
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(model=embedding_model_name)
        self.vector_store = None
        self.chunk_size = 500
        self.chunk_overlap = 100
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_or_create_db(self) -> Any:
        """Load existing vector DB or create new one"""
        try:
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            return self.vector_store
        except Exception as e:
            print(f"Error loading vector database: {e}")
            print("Creating new vector database...")
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.db_path
            )
            return self.vector_store
    
    def index_guideline(self, guideline: Dict) -> None:
        """Index a guideline into the vector database"""
        # Extract text and metadata
        texts = []
        metadatas = []
        
        # Basic guideline info
        guideline_text = f"Cancer Type: {guideline['cancer_type']}\n"
        guideline_text += f"Title: {guideline['title']}\n"
        guideline_text += f"Description: {guideline['description']}\n\n"
        
        # Process recommendations
        for i, rec in enumerate(guideline.get("recommendations", [])):
            rec_text = guideline_text  # Include basic info
            rec_text += f"Recommendation {i+1}: {rec['title']}\n"
            rec_text += f"Description: {rec['description']}\n"
            rec_text += f"Evidence Level: {rec['evidence_level']}\n"
            
            # Add detailed content
            if "detailed_content" in rec:
                rec_text += f"\nDetailed Content:\n{rec['detailed_content']}\n"
            
            # Add references
            if "references" in rec:
                rec_text += "\nReferences:\n"
                for ref in rec["references"]:
                    rec_text += f"- {ref}\n"
            
            texts.append(rec_text)
            metadatas.append({
                "cancer_type": guideline["cancer_type"],
                "recommendation_id": f"{guideline['id']}-rec-{i+1}",
                "evidence_level": rec["evidence_level"],
                "publication_date": guideline.get("publication_date"),
                "version": guideline.get("version"),
                "is_recommended_trial": "trial" in rec["title"].lower(),
                "genomic_target": self._extract_genomic_target(rec)
            })
        
        # Split into chunks
        docs = self.text_splitter.create_documents(texts, metadatas=metadatas)
        
        # Add to vector store
        self.vector_store.add_documents(docs)
    
    def _extract_genomic_target(self, recommendation: Dict) -> Optional[str]:
        """Extract genomic target from recommendation text"""
        # In a real implementation, use NLP to extract gene names
        # This is a simplified placeholder
        text = recommendation.get("title", "") + " " + recommendation.get("description", "")
        
        common_markers = ["EGFR", "ALK", "ROS1", "BRAF", "KRAS", "HER2", "PD-L1"]
        for marker in common_markers:
            if marker in text:
                return marker
        
        return None
    
    def search(self, query: str, filters: Optional[Dict] = None, k: int = 10) -> List[Dict]:
        """Search the vector database"""
        if not self.vector_store:
            self.load_or_create_db()
        
        docs = self.vector_store.similarity_search(
            query=query,
            filter=filters,
            k=k
        )
        
        return docs


class OncologyRAGReranker:
    """Specialized reranker for oncology recommendations"""
    
    def __init__(self, reranking_model_path: str, metadata_weights: Dict[str, float]):
        self.cross_encoder = CrossEncoder(reranking_model_path)
        self.metadata_weights = metadata_weights
    
    def rerank(self, query: str, patient_context: Dict, initial_results: List) -> List[Dict]:
        """
        Reranks initial retrieval results using specialized oncology relevance model
        """
        # Format patient context as text
        patient_text = self._format_patient_context(patient_context)
        
        # Create pairs for cross-encoder
        pairs = [(f"{query} [SEP] {patient_text}", doc.page_content) 
                for doc in initial_results]
        
        # Get semantic relevance scores
        semantic_scores = self.cross_encoder.predict(pairs)
        
        # Calculate metadata relevance
        metadata_scores = []
        for i, doc in enumerate(initial_results):
            score = 0
            
            # Evidence level weighting (A1 > A2 > B > C)
            if doc.metadata.get('evidence_level') == 'A1':
                score += self.metadata_weights['evidence_A1']
            elif doc.metadata.get('evidence_level') == 'A2':
                score += self.metadata_weights['evidence_A2']
            elif doc.metadata.get('evidence_level') == 'B':
                score += self.metadata_weights['evidence_B']
            
            # Recency weighting (more recent = higher score)
            pub_date = doc.metadata.get('publication_date')
            if pub_date:
                try:
                    pub_date = datetime.datetime.strptime(pub_date, "%Y-%m-%d")
                    days_old = (datetime.datetime.now() - pub_date).days
                    recency_score = min(1.0, max(0, 1 - (days_old / 365)))
                    score += recency_score * self.metadata_weights['recency']
                except (ValueError, TypeError):
                    pass
            
            # Clinical trial status weighting
            if doc.metadata.get('is_recommended_trial') == True:
                score += self.metadata_weights['recommended_trial']
            
            # Targeted therapy match
            if doc.metadata.get('genomic_target') in patient_context.get('biomarkers', []):
                score += self.metadata_weights['genomic_match']
            
            metadata_scores.append(score)
        
        # Combine scores (weighted sum of semantic and metadata relevance)
        combined_scores = [
            0.7 * semantic_scores[i] + 0.3 * metadata_scores[i]
            for i in range(len(initial_results))
        ]
        
        # Create reranked results
        reranked_results = [
            {
                "document": initial_results[i],
                "semantic_score": semantic_scores[i],
                "metadata_score": metadata_scores[i],
                "combined_score": combined_scores[i]
            }
            for i in range(len(initial_results))
        ]
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return reranked_results
    
    def _format_patient_context(self, patient_context: Dict) -> str:
        """Format patient context as text for the model"""
        text = f"Patient Information:\n"
        text += f"- Gender: {patient_context.get('patient', {}).get('gender', 'Unknown')}\n"
        text += f"- Age: {patient_context.get('patient', {}).get('age', 'Unknown')}\n\n"
        
        text += "Cancer Diagnosis:\n"
        for diagnosis in patient_context.get('cancer_diagnosis', []):
            text += f"- {diagnosis.get('code', 'Unknown cancer')}"
            if diagnosis.get('stage'):
                text += f", Stage: {diagnosis.get('stage')}"
            text += f", Onset: {diagnosis.get('onsetDate', 'Unknown')}\n"
        
        text += "\nBiomarkers:\n"
        for biomarker in patient_context.get('biomarkers', []):
            text += f"- {biomarker.get('code', 'Unknown')}: {biomarker.get('value', 'Unknown')}"
            if biomarker.get('interpretation'):
                text += f" ({biomarker.get('interpretation')})"
            text += "\n"
        
        text += "\nTreatment History:\n"
        for treatment in patient_context.get('treatment_history', []):
            text += f"- {treatment.get('medication', 'Unknown')}"
            if treatment.get('date'):
                text += f", Date: {treatment.get('date')}"
            if treatment.get('dosage'):
                text += f", Dose: {treatment.get('dosage')}"
            text += "\n"
        
        return text


class OncologyEventProcessor:
    """
    Processes real-time oncology data events from FHIR server
    """
    def __init__(self, fhir_subscription_endpoint: str, kafka_producer: Any):
        self.fhir_subscription_endpoint = fhir_subscription_endpoint
        self.kafka_producer = kafka_producer
        self.process_queue = asyncio.Queue()
        self.fhir_manager = FHIRIntegrationManager(CONFIG["fhir_server_url"])
    
    async def start_listening(self):
        """Start FHIR subscription listener for real-time updates"""
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.fhir_subscription_endpoint) as ws:
                print("Connected to FHIR subscription endpoint")
                
                # Process incoming messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self.process_queue.put(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
    
    async def process_events(self):
        """Process events from the queue"""
        while True:
            event_data = await self.process_queue.get()
            event = json.loads(event_data)
            
            # Extract event type and resource
            event_type = event.get('event', {}).get('code')
            resource_type = event.get('resource', {}).get('resourceType')
            resource_id = event.get('resource', {}).get('id')
            
            # Process based on resource type
            if resource_type == 'Observation' and self.is_oncology_relevant(event):
                # New lab result or biomarker
                await self.process_observation(event)
                
            elif resource_type == 'Condition' and self.is_cancer_condition(event):
                # New or updated cancer diagnosis
                await self.process_cancer_condition(event)
                
            elif resource_type == 'MedicationAdministration' and self.is_cancer_medication(event):
                # New cancer treatment administration
                await self.process_medication_administration(event)
                
            # Mark task as done
            self.process_queue.task_done()
    
    def is_oncology_relevant(self, event: Dict) -> bool:
        """Determine if an event is oncology-relevant"""
        resource = event.get('resource', {})
        resource_type = resource.get('resourceType')
        
        # Check LOINC codes for oncology relevance
        if resource_type == 'Observation':
            loinc_code = self.extract_loinc_code(resource)
            return loinc_code in ONCOLOGY_RELEVANT_LOINC_CODES
            
        # Check medication codes for chemotherapy agents
        if resource_type == 'MedicationAdministration':
            rxnorm_code = self.extract_rxnorm_code(resource)
            return rxnorm_code in CHEMOTHERAPY_RXNORM_CODES
            
        return False
    
    def is_cancer_condition(self, event: Dict) -> bool:
        """Check if condition is cancer-related"""
        resource = event.get('resource', {})
        return self.fhir_manager._is_cancer_condition(resource)
    
    def is_cancer_medication(self, event: Dict) -> bool:
        """Check if medication is cancer-related"""
        resource = event.get('resource', {})
        return self.fhir_manager._is_cancer_treatment(resource)
    
    def extract_loinc_code(self, resource: Dict) -> Optional[str]:
        """Extract LOINC code from resource"""
        return self.fhir_manager._extract_loinc_code(resource)
    
    def extract_rxnorm_code(self, resource: Dict) -> Optional[str]:
        """Extract RxNorm code from resource"""
        return self.fhir_manager._extract_rxnorm_code(resource)
    
    def extract_patient_id(self, resource: Dict) -> Optional[str]:
        """Extract patient ID from resource"""
        return resource.get('subject', {}).get('reference', '').replace('Patient/', '')
    
    async def process_observation(self, event: Dict):
        """Process oncology-relevant observation"""
        observation = event.get('resource', {})
        patient_id = self.extract_patient_id(observation)
        
        # Publish to Kafka for downstream processing
        await self.kafka_producer.send(
            'oncology-observations',
            key=patient_id,
            value=json.dumps({
                'patient_id': patient_id,
                'observation': observation,
                'timestamp': datetime.datetime.now().isoformat()
            })
        )
        
        # Check if this observation should trigger an alert
        if self.should_trigger_alert(observation):
            await self.generate_clinical_alert(patient_id, observation)
    
    async def process_cancer_condition(self, event: Dict):
        """Process cancer condition update"""
        condition = event.get('resource', {})
        patient_id = self.extract_patient_id(condition)
        
        # Publish to Kafka for downstream processing
        await self.kafka_producer.send(
            'oncology-conditions',
            key=patient_id,
            value=json.dumps({
                'patient_id': patient_id,
                'condition': condition,
                'timestamp': datetime.datetime.now().isoformat()
            })
        )
        
        # Trigger recommendation update
        await self.update_recommendations(patient_id)
    
    async def process_medication_administration(self, event: Dict):
        """Process medication administration"""
        medication = event.get('resource', {})
        patient_id = self.extract_patient_id(medication)
        
        # Publish to Kafka for downstream processing
        await self.kafka_producer.send(
            'oncology-medications',
            key=patient_id,
            value=json.dumps({
                'patient_id': patient_id,
                'medication': medication,
                'timestamp': datetime.datetime.now().isoformat()
            })
        )
        
        # Trigger recommendation update
        await self.update_recommendations(patient_id)
    
    def should_trigger_alert(self, observation: Dict) -> bool:
        """Determine if observation should trigger clinical alert"""
        # In a real implementation, would check against guideline thresholds
        # This is a simplified placeholder
        return False
    
    async def generate_clinical_alert(self, patient_id: str, observation: Dict):
        """Generate clinical alert based on observation"""
        # Get patient context
        patient_record = self.fhir_manager.get_patient_oncology_profile(patient_id)
        
        # Determine if alert is needed based on NCCN guidelines
        alert_data = self.check_guidelines_for_alert(patient_record, observation)
        
        if alert_data:
            # Send alert to appropriate channels
            await self.kafka_producer.send(
                'clinical-alerts',
                key=patient_id,
                value=json.dumps(alert_data)
            )
    
    def check_guidelines_for_alert(self, patient_record: Dict, observation: Dict) -> Optional[Dict]:
        """Check guidelines for alert conditions"""
        # In a real implementation, would query guidelines for alert thresholds
        # This is a simplified placeholder
        return None
    
    async def update_recommendations(self, patient_id: str):
        """Trigger update of patient recommendations"""
        await self.kafka_producer.send(
            'recommendation-updates',
            key=patient_id,
            value=json.dumps({
                'patient_id': patient_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'trigger': 'patient_data_change'
            })
        )
        print(f"Triggered recommendation update for patient {patient_id}")


class OncologyRAGOrchestrator:
    """Main orchestrator for the oncology decision support system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.fhir_manager = FHIRIntegrationManager(config["fhir_server_url"])
        self.nccn_api = NCCNGuidelinesAPI(config["nccn_api_key"], config["nccn_api_url"])
