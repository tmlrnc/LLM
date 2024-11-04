# medimate/
# ├── requirements.txt
# ├── config.py
# ├── main.py
# ├── knowledge_graph.py
# └── agent.py

# First, requirements.txt:
"""
langchain==0.0.284
openai==0.28.0
pinecone-client==2.2.2
neo4j==5.11.0
python-dotenv==1.0.0
numpy==1.24.3
"""

# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    
    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# knowledge_graph.py
from neo4j import GraphDatabase
from typing import Dict, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediMateKnowledgeGraph:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection and setup schema"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            logger.info("Successfully connected to Neo4j database")
            self.setup_schema()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def setup_schema(self):
        """Setup Neo4j schema and constraints"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) ON (p.id) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medication) ON (m.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Condition) ON (c.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) ON (s.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Provider) ON (pr.id) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Appointment) ON (a.id) IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.error(f"Failed to create constraint: {e}")
                    raise
    
    def add_patient(self, patient_data: Dict):
        """Add or update patient node"""
        query = """
        MERGE (p:Patient {id: $id})
        SET p += $properties
        RETURN p
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, {
                    "id": patient_data["id"],
                    "properties": patient_data
                })
                return result.single()["p"]
            except Exception as e:
                logger.error(f"Failed to add patient: {e}")
                raise
    
    def add_medical_entity(self, entity_type: str, entity_data: Dict):
        """Add medical entity (Condition, Medication, Symptom, etc.)"""
        query = f"""
        MERGE (e:{entity_type} {{name: $name}})
        SET e += $properties
        RETURN e
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, {
                    "name": entity_data["name"],
                    "properties": entity_data
                })
                return result.single()["e"]
            except Exception as e:
                logger.error(f"Failed to add {entity_type}: {e}")
                raise
    
    def create_medical_relationship(
        self,
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
        relationship: str,
        properties: Dict = None
    ):
        """Create relationship between medical entities"""
        query = f"""
        MATCH (a:{from_type} {{id: $from_id}})
        MATCH (b:{to_type} {{id: $to_id}})
        MERGE (a)-[r:{relationship}]->(b)
        SET r += $properties
        RETURN type(r) as relationship_type
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, {
                    "from_id": from_id,
                    "to_id": to_id,
                    "properties": properties or {}
                })
                return result.single()["relationship_type"]
            except Exception as e:
                logger.error(f"Failed to create relationship: {e}")
                raise
    
    def get_patient_medical_history(self, patient_id: str) -> Dict:
        """Retrieve complete medical history for a patient"""
        query = """
        MATCH (p:Patient {id: $patient_id})
        OPTIONAL MATCH (p)-[r]->(n)
        WITH p, collect(distinct {
            relationship: type(r),
            node_type: labels(n)[0],
            node_data: properties(n)
        }) as connections
        RETURN {
            patient: properties(p),
            connections: connections
        } as result
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, {"patient_id": patient_id})
                record = result.single()
                return record["result"] if record else None
            except Exception as e:
                logger.error(f"Failed to get patient history: {e}")
                raise
    
    def find_medication_interactions(self, medication_names: List[str]) -> List[Dict]:
        """Find interactions between medications"""
        query = """
        MATCH (m1:Medication)
        WHERE m1.name IN $medications
        MATCH (m1)-[r:INTERACTS_WITH]-(m2:Medication)
        WHERE m2.name IN $medications
        RETURN {
            medication1: m1.name,
            medication2: m2.name,
            severity: r.severity,
            description: r.description
        } as interaction
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, {"medications": medication_names})
                return [record["interaction"] for record in result]
            except Exception as e:
                logger.error(f"Failed to find medication interactions: {e}")
                raise
    
    def find_related_symptoms(self, condition_name: str) -> List[Dict]:
        """Find symptoms related to a condition"""
        query = """
        MATCH (c:Condition {name: $condition_name})-[:PRESENTS]->(s:Symptom)
        RETURN {
            symptom: s.name,
            description: s.description,
            severity: s.severity
        } as symptom_data
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, {"condition_name": condition_name})
                return [record["symptom_data"] for record in result]
            except Exception as e:
                logger.error(f"Failed to find related symptoms: {e}")
                raise

    def add_appointment(self, appointment_data: Dict):
        """Add medical appointment"""
        query = """
        MATCH (p:Patient {id: $patient_id})
        MATCH (pr:Provider {id: $provider_id})
        CREATE (a:Appointment {
            id: $appointment_id,
            date: $date,
            time: $time,
            type: $type,
            status: $status
        })
        CREATE (p)-[:HAS_APPOINTMENT]->(a)
        CREATE (a)-[:WITH_PROVIDER]->(pr)
        RETURN a
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, appointment_data)
                return result.single()["a"]
            except Exception as e:
                logger.error(f"Failed to add appointment: {e}")
                raise

# agent.py
from typing import List, Dict, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.agents import Tool, AgentExecutor, OpenAIFunctionsAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
import pinecone
from datetime import datetime
import numpy as np
from knowledge_graph import MediMateKnowledgeGraph

class MediMateAgent:
    def __init__(self, config: Config):
        """Initialize MediMate agent with configurations"""
        self.config = config
        self._initialize_components()
        self._initialize_knowledge_graph()
        self.agents = self._initialize_agents()
    
    def _initialize_components(self):
        """Initialize OpenAI and Pinecone components"""
        # Initialize Pinecone
        pinecone.init(
            api_key=self.config.PINECONE_API_KEY,
            environment=self.config.PINECONE_ENV
        )
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.OPENAI_API_KEY
        )
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            openai_api_key=self.config.OPENAI_API_KEY
        )
        
        # Initialize vector store
        self.index_name = "medimate-kb"
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                metric="cosine",
                dimension=1536
            )
        self.vectorstore = Pinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _initialize_knowledge_graph(self):
        """Initialize Neo4j knowledge graph"""
        self.knowledge_graph = MediMateKnowledgeGraph(
            uri=self.config.NEO4J_URI,
            username=self.config.NEO4J_USER,
            password=self.config.NEO4J_PASSWORD
        )
    
    def _initialize_agents(self) -> Dict:
        """Initialize all specialized agents"""
        return {
            "symptom_checker": self._create_symptom_checker_agent(),
            "medication_manager": self._create_medication_manager_agent(),
            "health_records": self._create_health_records_agent(),
            "appointment_scheduler": self._create_appointment_scheduler_agent()
        }
    
    def _create_base_tools(self) -> List[Tool]:
        """Create base tools available to all agents"""
        return [
            Tool(
                name="KnowledgeSearch",
                func=self.vectorstore.similarity_search,
                description="Search medical knowledge base"
            ),
            Tool(
                name="PatientHistory",
                func=self.knowledge_graph.get_patient_medical_history,
                description="Retrieve patient medical history"
            )
        ]
    
    def _create_symptom_checker_agent(self):
        """Create symptom checker specialized agent"""
        tools = self._create_base_tools() + [
            Tool(
                name="SymptomAnalysis",
                func=self.knowledge_graph.find_related_symptoms,
                description="Analyze symptoms and find related conditions"
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a medical symptom analysis expert. 
            Always ask clarifying questions about symptoms and provide evidence-based responses.
            If the situation seems urgent, recommend immediate medical attention."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        
        return OpenAIFunctionsAgent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
    
    # Add methods for other specialized agents...

    async def process_query(self, query: str, context: Dict = None) -> str:
        """Process user query using appropriate agent(s)"""
        try:
            # Determine relevant agents
            relevant_agents = self._get_relevant_agents(query)
            
            # Get knowledge context
            search_results = self.vectorstore.similarity_search(query, k=3)
            kb_context = "\n".join([doc.page_content for doc in search_results])
            
            # Get patient context if available
            patient_context = None
            if context and "patient_id" in context:
                patient_context = self.knowledge_graph.get_patient_medical_history(
                    context["patient_id"]
                )
            
            # Combine contexts
            full_context = {
                "knowledge_base": kb_context,
                "patient_history": patient_context
            }
            
            # Process with agents
            responses = []
            for agent_name in relevant_agents:
                agent = self.agents[agent_name]
                agent_executor = AgentExecutor.from_agent_and_tools(
                    agent=agent,
                    tools=agent.tools,
                    memory=self.memory,
                    verbose=True
                )
                
                response = await agent_executor.arun(
                    input=query,
                    context=full_context
                )
                responses.append(response)
            
            # Combine responses
            final_response = self._rerank_and_combine_responses(responses, query)
            
            # Update memory
            self.memory.save_context(
                {"input": query},
                {"output": final_response}
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I apologize, but I encountered an error processing your query. Please try again or rephrase your question."

# main.py
from config import Config
from agent import MediMateAgent
import asyncio

async def main():
    # Initialize configuration
    config = Config()
    
    # Initialize agent
    agent = MediMateAgent(config)
    
    # Example usage
    query = "What are the symptoms of the flu and when should I see a doctor?"
    context = {"patient_id": "12345"}  # Optional patient context
    
    response = await agent.process_query(query, context)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())