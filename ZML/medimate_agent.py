import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import pinecone
from typing import List, Dict, Optional
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()

    def scrape_webmd(self, url: str) -> Dict[str, str]:
        """Scrape health information from WebMD"""
        try:
            logger.info(f"Scraping URL: {url}")
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content
            content = {
                'title': '',
                'main_content': '',
                'symptoms': [],
                'treatments': []
            }
            
            # Get title
            title_elem = soup.find('h1')
            if title_elem:
                content['title'] = title_elem.text.strip()
            
            # Get main content
            article = soup.find('div', {'class': 'article-body'}) or soup.find('div', {'id': 'article-body'})
            if article:
                paragraphs = article.find_all('p')
                content['main_content'] = ' '.join([p.text.strip() for p in paragraphs])
            
            # Get lists
            lists = article.find_all('ul') if article else []
            for lst in lists:
                items = [li.text.strip() for li in lst.find_all('li')]
                if any('symptom' in item.lower() for item in items):
                    content['symptoms'].extend(items)
                if any('treat' in item.lower() for item in items):
                    content['treatments'].extend(items)
            
            return content
            
        except Exception as e:
            logger.error(f"Error scraping WebMD: {str(e)}")
            return {"error": str(e)}

class MediMateAgent:
    def __init__(self):
        logger.info("Initializing MediMate Agent...")
        
        # Initialize Pinecone
        pinecone.init(
            api_key="pcsk_2RN9ck_8N3q27K8tmgYDmfqct9U1VEdK48Ce4muTUkTtorQi1xeSBhikHsjtgZdjNkTh1h",
            environment="gcp-starter"
        )
        
        # Initialize components
        self.scraper = WebScraper()
        self.llm = Ollama(model="llama2")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create agents
        self.agents = self._initialize_agents()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Load initial data
        self._load_initial_data()
    
    def _initialize_agents(self) -> Dict:
        """Initialize all specialized agents"""
        try:
            return {
                'symptom_checker': self._create_agent('symptoms', self._get_symptom_prompt()),
                'medication_manager': self._create_agent('medications', self._get_medication_prompt()),
                'health_records': self._create_agent('records', self._get_records_prompt())
            }
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            return {}
    
    def _create_agent(self, name: str, prompt_template: str):
        """Create an agent with specified name and prompt"""
        vectorstore = self._create_vector_store(f"medimate-{name}")
        if not vectorstore:
            return None
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
            }
        )
    
    def _create_vector_store(self, index_name: str):
        """Create or get existing Pinecone index"""
        try:
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    metric="cosine",
                    dimension=384
                )
            return Pinecone.from_existing_index(
                index_name=index_name,
                embedding=self.embeddings
            )
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return None
    
    def _get_symptom_prompt(self) -> str:
        return """
        You are a medical symptom analyzer. Based on the context and query:
        1. Analyze the described symptoms
        2. Suggest possible conditions
        3. Indicate urgency level (Low/Medium/High)
        4. Recommend appropriate actions
        
        Context: {context}
        Question: {question}
        """
    
    def _get_medication_prompt(self) -> str:
        return """
        You are a medication management expert. Based on the context and query:
        1. Provide medication information
        2. Explain proper usage
        3. List potential side effects
        4. Note important warnings
        
        Context: {context}
        Question: {question}
        """
    
    def _get_records_prompt(self) -> str:
        return """
        You are a health records specialist. Based on the context and query:
        1. Summarize relevant health information
        2. Highlight important patterns
        3. Suggest follow-up actions
        4. Note areas needing attention
        
        Context: {context}
        Question: {question}
        """
    
    def _load_initial_data(self):
        """Load initial health data"""
        try:
            # Scrape WebMD flu symptoms page
            flu_info = self.scraper.scrape_webmd("https://www.webmd.com/cold-and-flu/adult-flu-symptoms")
            if "error" not in flu_info:
                self.add_knowledge([
                    flu_info['main_content'],
                    *flu_info.get('symptoms', []),
                    *flu_info.get('treatments', [])
                ])
        except Exception as e:
            logger.error(f"Error loading initial data: {str(e)}")
    
    def add_knowledge(self, texts: List[str]):
        """Add knowledge to all agents"""
        try:
            for agent_type, agent in self.agents.items():
                if agent:
                    vectorstore = self._create_vector_store(f"medimate-{agent_type}")
                    if vectorstore:
                        vectorstore.add_texts(texts)
            logger.info(f"Added {len(texts)} documents to knowledge base")
        except Exception as e:
            logger.error(f"Error adding knowledge: {str(e)}")
    
    def process_query(self, query: str) -> Dict:
        """Process user query"""
        try:
            # Determine agent type
            agent_type = self._determine_agent_type(query)
            agent = self.agents.get(agent_type)
            
            if not agent:
                return {"error": "No appropriate agent found"}
            
            # Process query
            response = agent({"query": query})
            
            return {
                "agent_type": agent_type,
                "answer": response["result"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"error": str(e)}
    
    def _determine_agent_type(self, query: str) -> str:
        """Determine appropriate agent for query"""
        query = query.lower()
        if any(word in query for word in ['symptom', 'feel', 'pain', 'sick']):
            return 'symptom_checker'
        elif any(word in query for word in ['medicine', 'medication', 'drug', 'pill']):
            return 'medication_manager'
        else:
            return 'health_records'

def main():
    try:
        # Initialize agent
        medimate = MediMateAgent()
        
        print("\n=== MediMate Health Assistant ===")
        print("Commands:")
        print("- 'quit': Exit program")
        print("- 'scrape [url]': Add health information from URL")
        print("Example questions:")
        print("- 'What are the symptoms of flu?'")
        print("- 'How should I take ibuprofen?'")
        print("- 'Can you check my health records?'\n")
        
        while True:
            query = input("\nHow can I help you today? ").strip()
            
            if query.lower() == 'quit':
                break
            
            if query.lower().startswith('scrape '):
                url = query[7:].strip()
                info = medimate.scraper.scrape_webmd(url)
                if "error" not in info:
                    medimate.add_knowledge([info['main_content']])
                    print("✅ Successfully added new health information")
                else:
                    print(f"❌ Error: {info['error']}")
                continue
            
            print("\nProcessing your query...")
            response = medimate.process_query(query)
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
            else:
                print(f"\n🤖 Agent: {response['agent_type']}")
                print(f"📝 Answer: {response['answer']}")
                print(f"⏰ Timestamp: {response['timestamp']}")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()