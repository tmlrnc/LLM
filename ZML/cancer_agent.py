#!/usr/bin/env python3
"""
Advanced Agentic Cancer Detection System - LangGraph Implementation
Multi-Agent Architecture with proper LangGraph coordination
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Annotated
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime
import uuid

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for system organization
class AgentType(Enum):
    MEDICAL_IMAGING = "medical_imaging"
    CLINICAL_REPORT = "clinical_report"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    TRIAL_MATCHING = "trial_matching"
    SUPERVISOR = "supervisor"

class CaseStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    NEEDS_REVIEW = "needs_review"
    COMPLETED = "completed"
    ESCALATED = "escalated"

# State definition for LangGraph
class CancerDetectionState(TypedDict):
    """State schema for the cancer detection workflow"""
    messages: Annotated[list, add_messages]
    patient_data: Dict[str, Any]
    imaging_findings: Optional[Dict[str, Any]]
    clinical_findings: Optional[Dict[str, Any]]
    knowledge_findings: Optional[Dict[str, Any]]
    trial_matches: Optional[List[Dict[str, Any]]]
    supervisor_decision: Optional[Dict[str, Any]]
    current_agent: Optional[str]
    confidence_scores: Dict[str, float]
    requires_human_review: bool
    session_id: str
    case_status: str

@dataclass
class PatientData:
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    current_symptoms: List[str]
    lab_results: Dict[str, float]
    imaging_data: Dict[str, Any]
    genomic_profile: Dict[str, str]

@dataclass
class ClinicalTrial:
    trial_id: str
    title: str
    phase: str
    eligibility_criteria: List[str]
    primary_endpoint: str
    location: str
    compatibility_score: float

class MedicalOntologyService:
    """Medical ontology service for SNOMED CT, UMLS integration"""
    
    def __init__(self):
        self.snomed_ct = {
            "chest_pain": "29857009",
            "lung_cancer": "363358000",
            "adenocarcinoma": "35917007",
            "stage_iv": "258215001"
        }
    
    def extract_medical_entities(self, text: str) -> List[Dict[str, str]]:
        entities = []
        for term, code in self.snomed_ct.items():
            if term.replace("_", " ") in text.lower():
                entities.append({
                    "term": term,
                    "snomed_code": code,
                    "confidence": 0.85
                })
        return entities

# Initialize the language model
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Define LangGraph tools
@tool
def analyze_medical_imaging(imaging_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze medical imaging data for tumor detection"""
    
    scan_type = imaging_data.get("type", "CT")
    scan_quality = imaging_data.get("quality", 0.9)
    
    # Simulated computer vision analysis
    findings = {
        "tumor_detected": True,
        "location": "right_upper_lobe",
        "size_mm": 32.5,
        "density_hu": 45,
        "enhancement_pattern": "heterogeneous",
        "lymph_nodes": ["mediastinal", "hilar"],
        "metastases": False,
        "stage_prediction": "T2N1M0",
        "confidence": min(scan_quality * 0.92, 0.95)
    }
    
    return findings

@tool
def analyze_clinical_reports(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze clinical reports and lab results using NLP"""
    
    ontology_service = MedicalOntologyService()
    
    # Process medical history and symptoms
    clinical_text = " ".join(
        patient_data.get("medical_history", []) + 
        patient_data.get("current_symptoms", [])
    )
    
    entities = ontology_service.extract_medical_entities(clinical_text)
    
    # Analyze lab results
    lab_results = patient_data.get("lab_results", {})
    lab_analysis = {}
    
    if "cea" in lab_results:
        cea_level = lab_results["cea"]
        lab_analysis["cea"] = "elevated" if cea_level > 5.0 else "normal"
    
    if "ldh" in lab_results:
        ldh_level = lab_results["ldh"]
        lab_analysis["ldh"] = "elevated" if ldh_level > 250 else "normal"
    
    # Extract biomarkers
    biomarkers = {}
    genomic_profile = patient_data.get("genomic_profile", {})
    for gene, mutation in genomic_profile.items():
        if gene in ["EGFR", "KRAS", "ALK", "PD-L1"]:
            biomarkers[gene] = mutation
    
    findings = {
        "extracted_entities": entities,
        "lab_analysis": lab_analysis,
        "biomarkers": biomarkers,
        "risk_factors": ["smoking_history"] if patient_data.get("age", 0) > 60 else [],
        "confidence": 0.88
    }
    
    return findings

@tool
def query_knowledge_graph(combined_findings: Dict[str, Any]) -> Dict[str, Any]:
    """Query medical knowledge graphs for evidence-based recommendations"""
    
    # Simulated knowledge graph with treatment recommendations
    medical_kg = {
        "lung_cancer": {
            "treatments": [
                {"name": "surgery", "stage_applicability": ["I", "II", "IIIA"], "success_rate": 0.75},
                {"name": "immunotherapy", "stage_applicability": ["III", "IV"], "success_rate": 0.45},
                {"name": "targeted_therapy", "stage_applicability": ["III", "IV"], "success_rate": 0.65}
            ],
            "prognosis": {"5_year_survival": 0.61}
        }
    }
    
    findings = {
        "treatment_options": medical_kg["lung_cancer"]["treatments"],
        "drug_interactions": ["Monitor for skin toxicity with EGFR inhibitors"],
        "prognosis": medical_kg["lung_cancer"]["prognosis"],
        "evidence_level": "high",
        "literature_references": ["NEJM 2023;389:123-135", "J Clin Oncol 2023;41:234-246"],
        "confidence": 0.91
    }
    
    return findings

@tool
def match_clinical_trials(patient_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Match patient to relevant clinical trials"""
    
    # Mock clinical trials database
    trials = [
        {
            "trial_id": "NCT05123456",
            "title": "Phase II Study of Novel Immunotherapy in Advanced Lung Cancer",
            "phase": "II",
            "eligibility_criteria": [
                "Stage IIIB-IV lung adenocarcinoma",
                "PD-L1 expression >1%",
                "Age 18-75"
            ],
            "location": "Memorial Sloan Kettering Cancer Center",
            "compatibility_score": 0.85
        },
        {
            "trial_id": "NCT05234567",
            "title": "Targeted Therapy for EGFR-Mutated Lung Cancer",
            "phase": "III",
            "eligibility_criteria": [
                "EGFR-mutated lung cancer",
                "Previously untreated",
                "Stage IIIB-IV"
            ],
            "location": "MD Anderson Cancer Center",
            "compatibility_score": 0.92
        }
    ]
    
    # Filter trials based on patient compatibility
    matched_trials = [trial for trial in trials if trial["compatibility_score"] > 0.5]
    matched_trials.sort(key=lambda x: x["compatibility_score"], reverse=True)
    
    return matched_trials[:5]  # Top 5 matches

# LangGraph Node Functions
async def medical_imaging_node(state: CancerDetectionState) -> CancerDetectionState:
    """Medical imaging analysis node"""
    logger.info("Processing medical imaging analysis...")
    
    # Create AI message with imaging analysis prompt
    imaging_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical imaging AI specialist. Analyze the provided imaging data 
        and provide detailed findings about potential tumors, their characteristics, and staging information.
        Use the analyze_medical_imaging tool to process the scan data."""),
        ("human", "Please analyze the medical imaging data: {imaging_data}")
    ])
    
    # Format the prompt with patient data
    imaging_data = state["patient_data"].get("imaging_data", {})
    
    # Use the tool to analyze imaging
    imaging_findings = analyze_medical_imaging.invoke(imaging_data)
    
    # Create response message
    analysis_message = AIMessage(
        content=f"Medical imaging analysis complete. Tumor detected: {imaging_findings.get('tumor_detected')}. "
                f"Location: {imaging_findings.get('location')}. "
                f"Predicted stage: {imaging_findings.get('stage_prediction')}."
    )
    
    return {
        **state,
        "messages": [analysis_message],
        "imaging_findings": imaging_findings,
        "confidence_scores": {**state.get("confidence_scores", {}), "imaging": imaging_findings.get("confidence", 0.0)},
        "current_agent": "medical_imaging"
    }

async def clinical_report_node(state: CancerDetectionState) -> CancerDetectionState:
    """Clinical report analysis node"""
    logger.info("Processing clinical report analysis...")
    
    clinical_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a clinical data analysis specialist. Process patient medical history, 
        symptoms, lab results, and genomic profiles to extract relevant medical entities and insights.
        Use the analyze_clinical_reports tool for processing."""),
        ("human", "Please analyze the clinical data: {patient_data}")
    ])
    
    # Use the tool to analyze clinical data
    clinical_findings = analyze_clinical_reports.invoke(state["patient_data"])
    
    # Create response message
    analysis_message = AIMessage(
        content=f"Clinical analysis complete. Entities extracted: {len(clinical_findings.get('extracted_entities', []))}. "
                f"Biomarkers identified: {list(clinical_findings.get('biomarkers', {}).keys())}. "
                f"Risk factors: {clinical_findings.get('risk_factors', [])}."
    )
    
    return {
        **state,
        "messages": [analysis_message],
        "clinical_findings": clinical_findings,
        "confidence_scores": {**state.get("confidence_scores", {}), "clinical": clinical_findings.get("confidence", 0.0)},
        "current_agent": "clinical_report"
    }

async def knowledge_graph_node(state: CancerDetectionState) -> CancerDetectionState:
    """Knowledge graph query node"""
    logger.info("Querying medical knowledge graph...")
    
    # Combine findings from previous agents
    combined_findings = {
        **(state.get("imaging_findings", {})),
        **(state.get("clinical_findings", {}))
    }
    
    # Query knowledge graph
    kg_findings = query_knowledge_graph.invoke(combined_findings)
    
    analysis_message = AIMessage(
        content=f"Knowledge graph analysis complete. Treatment options identified: {len(kg_findings.get('treatment_options', []))}. "
                f"Evidence level: {kg_findings.get('evidence_level')}. "
                f"Prognosis data available: {'Yes' if kg_findings.get('prognosis') else 'No'}."
    )
    
    return {
        **state,
        "messages": [analysis_message],
        "knowledge_findings": kg_findings,
        "confidence_scores": {**state.get("confidence_scores", {}), "knowledge": kg_findings.get("confidence", 0.0)},
        "current_agent": "knowledge_graph"
    }

async def trial_matching_node(state: CancerDetectionState) -> CancerDetectionState:
    """Clinical trial matching node"""
    logger.info("Matching clinical trials...")
    
    # Create patient profile for trial matching
    patient_profile = {
        "age": state["patient_data"].get("age"),
        "gender": state["patient_data"].get("gender"),
        "disease_stage": state.get("imaging_findings", {}).get("stage_prediction"),
        "biomarkers": state.get("clinical_findings", {}).get("biomarkers", {}),
        "medical_history": state["patient_data"].get("medical_history", [])
    }
    
    # Match trials
    matched_trials = match_clinical_trials.invoke(patient_profile)
    
    analysis_message = AIMessage(
        content=f"Clinical trial matching complete. Found {len(matched_trials)} suitable trials. "
                f"Top match: {matched_trials[0]['title'] if matched_trials else 'None'} "
                f"(compatibility: {matched_trials[0]['compatibility_score']:.1%} if matched_trials else 'N/A')."
    )
    
    return {
        **state,
        "messages": [analysis_message],
        "trial_matches": matched_trials,
        "confidence_scores": {**state.get("confidence_scores", {}), "trials": 0.87},
        "current_agent": "trial_matching"
    }

async def supervisor_node(state: CancerDetectionState) -> CancerDetectionState:
    """Supervisor node for human-in-the-loop decision making"""
    logger.info("Supervisor reviewing case...")
    
    # Analyze all agent responses for consensus
    confidence_scores = state.get("confidence_scores", {})
    avg_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.0
    
    # Determine if human review is needed
    needs_review = (
        avg_confidence < 0.85 or
        len(confidence_scores) < 4 or  # Not all agents completed
        any(score < 0.8 for score in confidence_scores.values())
    )
    
    # Create supervisor decision
    supervisor_decision = {
        "avg_confidence": avg_confidence,
        "consensus_achieved": avg_confidence > 0.85,
        "needs_human_review": needs_review,
        "case_complexity": "high" if needs_review else "medium",
        "escalation_reason": "Low confidence scores" if needs_review else None,
        "processing_complete": True
    }
    
    if needs_review:
        status = CaseStatus.NEEDS_REVIEW.value
        summary_msg = f"Case requires human review. Average confidence: {avg_confidence:.1%}. Escalating to specialist."
    else:
        status = CaseStatus.COMPLETED.value
        summary_msg = f"Case analysis complete. Average confidence: {avg_confidence:.1%}. Ready for treatment planning."
    
    analysis_message = AIMessage(content=summary_msg)
    
    return {
        **state,
        "messages": [analysis_message],
        "supervisor_decision": supervisor_decision,
        "requires_human_review": needs_review,
        "case_status": status,
        "current_agent": "supervisor"
    }

def route_next_step(state: CancerDetectionState) -> str:
    """Router function to determine the next step in the workflow"""
    current_agent = state.get("current_agent")
    
    # Define the workflow sequence
    if current_agent is None:
        return "medical_imaging"
    elif current_agent == "medical_imaging":
        return "clinical_report"
    elif current_agent == "clinical_report":
        return "knowledge_graph"
    elif current_agent == "knowledge_graph":
        return "trial_matching"
    elif current_agent == "trial_matching":
        return "supervisor"
    elif current_agent == "supervisor":
        return END
    else:
        return END

def should_continue(state: CancerDetectionState) -> str:
    """Conditional edge function to determine if workflow should continue"""
    if state.get("current_agent") == "supervisor":
        return END
    else:
        return route_next_step(state)

class CancerDetectionWorkflow:
    """Main workflow coordinator using LangGraph"""
    
    def __init__(self):
        self.graph = self._build_graph()
        self.session_id = str(uuid.uuid4())
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the state graph
        workflow = StateGraph(CancerDetectionState)
        
        # Add all agent nodes
        workflow.add_node("medical_imaging", medical_imaging_node)
        workflow.add_node("clinical_report", clinical_report_node)
        workflow.add_node("knowledge_graph", knowledge_graph_node)
        workflow.add_node("trial_matching", trial_matching_node)
        workflow.add_node("supervisor", supervisor_node)
        
        # Define the workflow edges
        workflow.add_edge(START, "medical_imaging")
        workflow.add_edge("medical_imaging", "clinical_report")
        workflow.add_edge("clinical_report", "knowledge_graph") 
        workflow.add_edge("knowledge_graph", "trial_matching")
        workflow.add_edge("trial_matching", "supervisor")
        workflow.add_edge("supervisor", END)
        
        # Compile the graph
        return workflow.compile()
    
    async def process_patient_case(self, patient_data: PatientData) -> Dict[str, Any]:
        """Process a patient case through the multi-agent workflow"""
        logger.info(f"Starting case processing for patient {patient_data.patient_id}")
        
        case_start_time = datetime.now()
        
        try:
            # Initialize the state
            initial_state = CancerDetectionState(
                messages=[HumanMessage(content=f"Analyze patient case {patient_data.patient_id}")],
                patient_data=asdict(patient_data),
                imaging_findings=None,
                clinical_findings=None,
                knowledge_findings=None,
                trial_matches=None,
                supervisor_decision=None,
                current_agent=None,
                confidence_scores={},
                requires_human_review=False,
                session_id=self.session_id,
                case_status=CaseStatus.IN_PROGRESS.value
            )
            
            # Run the workflow
            logger.info("Executing LangGraph workflow...")
            final_state = await self.graph.ainvoke(initial_state)
            
            # Compile results
            total_processing_time = (datetime.now() - case_start_time).total_seconds()
            
            case_result = {
                "session_id": self.session_id,
                "patient_id": patient_data.patient_id,
                "processing_time": total_processing_time,
                "status": final_state.get("case_status", CaseStatus.COMPLETED.value),
                "findings": {
                    "imaging": final_state.get("imaging_findings"),
                    "clinical": final_state.get("clinical_findings"),
                    "knowledge_graph": final_state.get("knowledge_findings"),
                    "trial_matches": final_state.get("trial_matches")
                },
                "supervisor_decision": final_state.get("supervisor_decision"),
                "confidence_metrics": self._calculate_confidence_metrics(final_state),
                "summary": self._generate_case_summary(final_state),
                "next_steps": self._determine_next_steps(final_state),
                "workflow_messages": [msg.content for msg in final_state.get("messages", [])]
            }
            
            logger.info(f"Case processing completed in {total_processing_time:.2f} seconds")
            return case_result
            
        except Exception as e:
            logger.error(f"Error processing patient case: {str(e)}")
            return {
                "session_id": self.session_id,
                "patient_id": patient_data.patient_id,
                "status": CaseStatus.ESCALATED.value,
                "error": str(e),
                "processing_time": (datetime.now() - case_start_time).total_seconds()
            }
    
    def _calculate_confidence_metrics(self, state: CancerDetectionState) -> Dict[str, float]:
        """Calculate overall confidence metrics"""
        confidence_scores = state.get("confidence_scores", {})
        
        if not confidence_scores:
            return {"overall_confidence": 0.0}
        
        values = list(confidence_scores.values())
        return {
            "overall_confidence": np.mean(values),
            "confidence_range": [np.min(values), np.max(values)],
            "standard_deviation": np.std(values),
            "agents_above_threshold": sum(1 for c in values if c > 0.85) / len(values)
        }
    
    def _generate_case_summary(self, state: CancerDetectionState) -> Dict[str, Any]:
        """Generate comprehensive case summary"""
        imaging = state.get("imaging_findings", {})
        clinical = state.get("clinical_findings", {})
        
        return {
            "primary_diagnosis": f"Lung adenocarcinoma, {imaging.get('stage_prediction', 'stage unknown')}",
            "key_findings": [
                f"Tumor location: {imaging.get('location', 'unknown')}",
                f"Size: {imaging.get('size_mm', 'unknown')}mm",
                f"Biomarkers: {list(clinical.get('biomarkers', {}).keys())}"
            ],
            "treatment_urgency": "moderate",
            "prognosis": "favorable with treatment"
        }
    
    def _determine_next_steps(self, state: CancerDetectionState) -> List[str]:
        """Determine next steps based on analysis"""
        if state.get("requires_human_review"):
            return [
                "Schedule urgent oncology consultation",
                "Prepare case for multidisciplinary team review",
                "Obtain additional imaging if needed",
                "Discuss findings with patient and family"
            ]
        else:
            return [
                "Proceed with recommended treatment plan",
                "Schedule follow-up appointments",
                "Begin treatment coordination",
                "Monitor patient response"
            ]

class StreamingChatInterface:
    """Interactive chat interface with streaming support"""
    
    def __init__(self):
        self.workflow = CancerDetectionWorkflow()
    
    async def run_interactive_session(self):
        """Run an interactive session with the cancer detection system"""
        print("🏥 Advanced Cancer Detection System - LangGraph Implementation")
        print("=" * 70)
        print("This system uses multiple AI agents coordinated by LangGraph.")
        print("Type 'process_case' to analyze a sample patient, or 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("User: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                elif user_input.lower() == "process_case":
                    await self._process_sample_case()
                else:
                    print("Available commands: 'process_case', 'quit'")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def _process_sample_case(self):
        """Process a sample patient case"""
        
        # Create sample patient
        sample_patient = PatientData(
            patient_id="P123456",
            age=67,
            gender="male",
            medical_history=[
                "smoking history 30 pack-years",
                "hypertension", 
                "diabetes type 2"
            ],
            current_symptoms=[
                "persistent cough",
                "chest pain",
                "weight loss",
                "shortness of breath"
            ],
            lab_results={
                "cea": 8.5,  # elevated
                "ldh": 320,  # elevated
                "hemoglobin": 11.2
            },
            imaging_data={
                "type": "CT_chest",
                "quality": 0.92,
                "contrast": True
            },
            genomic_profile={
                "EGFR": "L858R_mutation",
                "KRAS": "wild_type",
                "PD-L1": "50_percent_positive"
            }
        )
        
        print(f"\n🔄 Processing patient {sample_patient.patient_id}...")
        print(f"Patient: {sample_patient.age}yr {sample_patient.gender}")
        print(f"Symptoms: {', '.join(sample_patient.current_symptoms[:2])}...")
        
        # Process through LangGraph workflow
        result = await self.workflow.process_patient_case(sample_patient)
        
        # Display results
        self._display_results(result)
    
    def _display_results(self, result: Dict[str, Any]):
        """Display case analysis results"""
        print("\n📊 CASE ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Status: {result['status'].upper()}")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print(f"Overall Confidence: {result['confidence_metrics'].get('overall_confidence', 0):.1%}")
        
        print("\n🔍 WORKFLOW PROGRESS:")
        for msg in result.get("workflow_messages", []):
            print(f"  • {msg}")
        
        summary = result.get("summary", {})
        print(f"\n📋 DIAGNOSIS: {summary.get('primary_diagnosis', 'Unknown')}")
        print(f"Treatment Urgency: {summary.get('treatment_urgency', 'Unknown').title()}")
        
        print(f"\n🎯 NEXT STEPS:")
        for i, step in enumerate(result.get("next_steps", []), 1):
            print(f"  {i}. {step}")
        
        if result.get("status") == CaseStatus.NEEDS_REVIEW.value:
            print("\n⚠️  HUMAN REVIEW REQUIRED")
            supervisor = result.get("supervisor_decision", {})
            if supervisor.get("escalation_reason"):
                print(f"Reason: {supervisor['escalation_reason']}")

async def main():
    """Main entry point"""
    interface = StreamingChatInterface()
    await interface.run_interactive_session()

if __name__ == "__main__":
    asyncio.run(main())