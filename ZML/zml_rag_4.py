import os
from typing import List, Dict
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

class MediMateAgent:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_env: str):
        # Initialize API keys and environment
        self.openai_api_key = openai_api_key
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            openai_api_key=openai_api_key
        )
        
        # Initialize vector store
        self.index_name = "medimate-kb"
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                metric="cosine",
                dimension=1536  # OpenAI embedding dimension
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
        
        # Initialize specialized agents
        self.agents = self._initialize_agents()
        
    def _initialize_agents(self) -> Dict:
        """Initialize specialized agents for different healthcare domains"""
        agents = {
            "symptom_checker": self._create_symptom_checker_agent(),
            "medication_manager": self._create_medication_manager_agent(),
            "health_records": self._create_health_records_agent(),
            "appointment_scheduler": self._create_appointment_scheduler_agent(),
            "expense_tracker": self._create_expense_tracker_agent()
        }
        return agents
    
    def _create_symptom_checker_agent(self):
        """Create specialized agent for symptom checking"""
        tools = [
            Tool(
                name="SymptomAnalysis",
                func=self._analyze_symptoms,
                description="Analyzes reported symptoms and provides possible conditions"
            ),
            Tool(
                name="TriageAssessment",
                func=self._assess_urgency,
                description="Assesses the urgency of medical attention needed"
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a medical symptom analysis expert. 
            Always ask clarifying questions about symptoms and provide evidence-based responses.
            If the situation seems urgent, recommend immediate medical attention."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
        ])
        
        return OpenAIFunctionsAgent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
    
    def _create_medication_manager_agent(self):
        """Create specialized agent for medication management"""
        tools = [
            Tool(
                name="MedicationInfo",
                func=self._get_medication_info,
                description="Provides information about medications"
            ),
            Tool(
                name="InteractionCheck",
                func=self._check_interactions,
                description="Checks for potential medication interactions"
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a medication management expert.
            Provide accurate medication information and always check for interactions.
            Remind users to consult healthcare providers for medical advice."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
        ])
        
        return OpenAIFunctionsAgent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
    
    def _create_health_records_agent(self):
        """Create specialized agent for health records management"""
        tools = [
            Tool(
                name="RecordRetrieval",
                func=self._retrieve_health_records,
                description="Retrieves and summarizes health records"
            ),
            Tool(
                name="TrendAnalysis",
                func=self._analyze_health_trends,
                description="Analyzes health trends from historical data"
            )
        ]
        
        return OpenAIFunctionsAgent(
            llm=self.llm,
            tools=tools,
            prompt=self._get_health_records_prompt()
        )
    
    def _get_health_records_prompt(self):
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a health records specialist.
            Provide clear summaries of health records and identify important trends.
            Maintain strict confidentiality and privacy of health information."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
        ])
    
    async def process_query(self, query: str, context: Dict = None) -> str:
        """Process user query using appropriate agent(s)"""
        # Determine relevant agent(s) based on query intent
        relevant_agents = self._get_relevant_agents(query)
        
        # Retrieve relevant context from vector store
        search_results = self.vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in search_results])
        
        # Process query with each relevant agent
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
                context=context
            )
            responses.append(response)
        
        # Rerank and combine responses
        final_response = self._rerank_and_combine_responses(responses, query)
        
        # Update conversation memory
        self.memory.save_context({"input": query}, {"output": final_response})
        
        return final_response
    
    def _get_relevant_agents(self, query: str) -> List[str]:
        """Determine which agents are relevant for the given query"""
        # Embed query and compare with agent specialties
        query_embedding = self.embeddings.embed_query(query)
        
        # Define agent specialties and their embeddings
        agent_specialties = {
            "symptom_checker": "symptoms medical conditions health assessment",
            "medication_manager": "medications drugs prescriptions interactions",
            "health_records": "medical records history lab results",
            "appointment_scheduler": "appointments scheduling bookings",
            "expense_tracker": "medical expenses costs bills insurance"
        }
        
        # Calculate relevance scores
        relevance_scores = {}
        for agent_name, specialty in agent_specialties.items():
            specialty_embedding = self.embeddings.embed_query(specialty)
            similarity = np.dot(query_embedding, specialty_embedding)
            relevance_scores[agent_name] = similarity
        
        # Select agents above threshold
        threshold = 0.7
        relevant_agents = [
            agent for agent, score in relevance_scores.items()
            if score > threshold
        ]
        
        return relevant_agents or ["general"]  # Default to general if no specific agents are relevant
    
    def _rerank_and_combine_responses(self, responses: List[str], query: str) -> str:
        """Rerank and combine responses based on relevance and completeness"""
        if not responses:
            return "I apologize, but I couldn't generate a relevant response. Please try rephrasing your question."
        
        # Create a prompt for the LLM to combine and improve responses
        combination_prompt = f"""
        Original question: {query}
        
        Multiple perspectives have been provided:
        {' '.join(responses)}
        
        Please synthesize these perspectives into a single, comprehensive response that:
        1. Directly answers the original question
        2. Includes the most relevant information from all perspectives
        3. Is coherent and well-organized
        4. Adds any missing critical information
        5. Maintains a professional and helpful tone
        """
        
        final_response = self.llm.predict(combination_prompt)
        return final_response

    # Additional helper methods for specific tools
    def _analyze_symptoms(self, symptoms: str) -> str:
        """Analyze symptoms and provide possible conditions"""
        # Implementation for symptom analysis
        pass

    def _assess_urgency(self, symptoms: str) -> str:
        """Assess the urgency of medical attention needed"""
        # Implementation for urgency assessment
        pass

    def _get_medication_info(self, medication: str) -> str:
        """Get information about medications"""
        # Implementation for medication information retrieval
        pass

    def _check_interactions(self, medications: List[str]) -> str:
        """Check for potential medication interactions"""
        # Implementation for interaction checking
        pass
```

To use this system:

1. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_ENV="your-pinecone-environment"
```

2. Create a usage script:
```python
async def main():
    # Initialize the agent
    agent = MediMateAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_env=os.getenv("PINECONE_ENV")
    )
    
    # Example query
    query = "What are the symptoms of the flu and when should I see a doctor?"
    response = await agent.process_query(query)
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

