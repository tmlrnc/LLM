"""
RAG-Enhanced Cognitive Load Prediction System with LangChain Integration
Combines knowledge graphs, multi-agent reasoning, and LangChain tools for improved accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import random
import asyncio

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated
import operator

@dataclass
class UserProfile:
    user_id: str
    demographics: Dict[str, Any] = field(default_factory=dict)
    cognitive_traits: Dict[str, float] = field(default_factory=dict)
    historical_performance: Dict[str, float] = field(default_factory=dict)
    expertise_level: str = "beginner"

@dataclass
class TaskContext:
    task_type: str
    task_domain: str = "general"
    complexity_level: float = 0.5
    duration: float = 300.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()

@dataclass
class CognitivePattern:
    pattern_id: str
    features: np.ndarray
    cognitive_load_class: int
    user_context: UserProfile
    task_context: TaskContext
    confidence: float = 0.8
    created_at: str = ""
    reasoning: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.datetime.now().isoformat()

# LangGraph State Definition
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], operator.add]
    user_profile: Optional[UserProfile]
    task_context: Optional[TaskContext]
    features: Optional[np.ndarray]
    predictions: Dict[str, Any]
    retrieved_patterns: List[CognitivePattern]
    final_prediction: Optional[Dict[str, Any]]
    reasoning_chain: List[str]

class EnhancedKnowledgeGraph:
    """Advanced Knowledge Graph with LangChain integration"""
    
    def __init__(self, llm_model=None):
        self.user_patterns: Dict[str, List[CognitivePattern]] = {}
        self.task_patterns: Dict[str, List[CognitivePattern]] = {}
        self.transition_patterns: Dict[str, List[Dict]] = {}
        self.feature_importance: Dict[str, float] = {}
        self.scaler = StandardScaler()
        self._is_fitted = False
        
        # LangChain components
        self.llm = llm_model or init_chat_model("anthropic:claude-3-5-sonnet-latest")
        self.graph_transformer = LLMGraphTransformer(llm=self.llm) if llm_model else None
        
        # Semantic search components
        self.pattern_embeddings: Dict[str, np.ndarray] = {}
        self.context_embeddings: Dict[str, np.ndarray] = {}
        
    async def add_pattern_with_reasoning(self, pattern: CognitivePattern, reasoning: str = ""):
        """Add pattern with LLM-generated reasoning"""
        pattern.reasoning = reasoning or await self._generate_pattern_reasoning(pattern)
        self.add_pattern(pattern)
        
        # Generate embeddings for semantic search
        await self._update_pattern_embeddings(pattern)
    
    async def _generate_pattern_reasoning(self, pattern: CognitivePattern) -> str:
        """Generate reasoning for why this pattern indicates certain cognitive load"""
        if not self.llm:
            return "Pattern added without reasoning analysis"
            
        prompt = ChatPromptTemplate.from_template(
            """Analyze this cognitive load pattern and explain the reasoning:
            
            User Profile: {user_profile}
            Task Context: {task_context}
            Cognitive Load Class: {cognitive_load} (0=Low, 1=Moderate, 2=High)
            Feature Summary: {features_summary}
            
            Provide a brief explanation of why this combination of factors leads to the predicted cognitive load level.
            Focus on the relationships between user expertise, task complexity, and physiological indicators."""
        )
        
        features_summary = self._summarize_features(pattern.features)
        
        messages = prompt.format_messages(
            user_profile=f"Expertise: {pattern.user_context.expertise_level}, Traits: {pattern.user_context.cognitive_traits}",
            task_context=f"Type: {pattern.task_context.task_type}, Complexity: {pattern.task_context.complexity_level}",
            cognitive_load=pattern.cognitive_load_class,
            features_summary=features_summary
        )
        
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def _summarize_features(self, features: np.ndarray) -> str:
        """Create human-readable summary of physiological features"""
        # Mock feature interpretation - in practice, map to actual physiological metrics
        avg_activation = np.mean(features)
        variability = np.std(features)
        
        if avg_activation > 0.7:
            activation_level = "high"
        elif avg_activation > 0.4:
            activation_level = "moderate"
        else:
            activation_level = "low"
            
        return f"Average activation: {activation_level}, Variability: {variability:.2f}"
    
    async def _update_pattern_embeddings(self, pattern: CognitivePattern):
        """Update semantic embeddings for pattern retrieval"""
        # Create text representation for embedding
        pattern_text = f"""
        User: {pattern.user_context.expertise_level} expertise
        Task: {pattern.task_context.task_type} complexity {pattern.task_context.complexity_level}
        Load: {pattern.cognitive_load_class}
        Reasoning: {pattern.reasoning}
        """
        
        # In a real implementation, use actual embedding model
        # For demo, create mock embedding based on pattern characteristics
        embedding = np.random.rand(384)  # Mock 384-dim embedding
        self.pattern_embeddings[pattern.pattern_id] = embedding
    
    def add_pattern(self, pattern: CognitivePattern):
        """Add cognitive load pattern to knowledge graph"""
        # Store by user
        if pattern.user_context.user_id not in self.user_patterns:
            self.user_patterns[pattern.user_context.user_id] = []
        self.user_patterns[pattern.user_context.user_id].append(pattern)
        
        # Store by task type
        task_key = f"{pattern.task_context.task_type}_{pattern.task_context.complexity_level:.1f}"
        if task_key not in self.task_patterns:
            self.task_patterns[task_key] = []
        self.task_patterns[task_key].append(pattern)
        
        # Update feature scaler if we have enough patterns
        if len(self._get_all_patterns()) >= 10 and not self._is_fitted:
            self._fit_scaler()
    
    def _get_all_patterns(self) -> List[CognitivePattern]:
        """Get all patterns from the knowledge graph"""
        all_patterns = []
        for patterns in self.user_patterns.values():
            all_patterns.extend(patterns)
        return all_patterns
    
    def _fit_scaler(self):
        """Fit the feature scaler on all available patterns"""
        all_patterns = self._get_all_patterns()
        if len(all_patterns) < 2:
            return
            
        feature_matrix = np.vstack([p.features for p in all_patterns])
        self.scaler.fit(feature_matrix)
        self._is_fitted = True
    
    async def retrieve_similar_patterns_semantic(self, 
                                               query_features: np.ndarray,
                                               user_profile: UserProfile,
                                               task_context: TaskContext,
                                               similarity_threshold: float = 0.6) -> List[CognitivePattern]:
        """Enhanced pattern retrieval using semantic similarity"""
        # Generate query embedding
        query_text = f"""
        User: {user_profile.expertise_level} expertise
        Task: {task_context.task_type} complexity {task_context.complexity_level}
        """
        
        # Mock query embedding
        query_embedding = np.random.rand(384)
        
        # Find similar patterns using both feature and semantic similarity
        similar_patterns = []
        
        for pattern in self._get_all_patterns():
            if pattern.pattern_id in self.pattern_embeddings:
                # Semantic similarity
                semantic_sim = self._calculate_similarity(
                    query_embedding, 
                    self.pattern_embeddings[pattern.pattern_id]
                )
                
                # Feature similarity
                feature_sim = self._calculate_similarity(query_features, pattern.features)
                
                # Task similarity
                task_sim = self._task_similarity(task_context, pattern.task_context)
                
                # Combined similarity score
                combined_sim = (semantic_sim * 0.4 + feature_sim * 0.4 + task_sim * 0.2)
                
                if combined_sim > similarity_threshold:
                    adjusted_pattern = CognitivePattern(
                        pattern_id=pattern.pattern_id,
                        features=pattern.features,
                        cognitive_load_class=pattern.cognitive_load_class,
                        user_context=pattern.user_context,
                        task_context=pattern.task_context,
                        confidence=pattern.confidence * combined_sim,
                        created_at=pattern.created_at,
                        reasoning=pattern.reasoning
                    )
                    similar_patterns.append(adjusted_pattern)
        
        return sorted(similar_patterns, key=lambda p: p.confidence, reverse=True)[:10]
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors"""
        try:
            f1 = features1.flatten()
            f2 = features2.flatten()
            
            min_len = min(len(f1), len(f2))
            f1 = f1[:min_len]
            f2 = f2[:min_len]
            
            if len(f1) == 0 or len(f2) == 0:
                return 0.0
            
            dot_product = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def _task_similarity(self, task1: TaskContext, task2: TaskContext) -> float:
        """Calculate similarity between task contexts"""
        similarities = []
        
        type_sim = 1.0 if task1.task_type == task2.task_type else 0.3
        similarities.append(type_sim)
        
        domain_sim = 1.0 if task1.task_domain == task2.task_domain else 0.4
        similarities.append(domain_sim)
        
        complexity_diff = abs(task1.complexity_level - task2.complexity_level)
        complexity_sim = max(0.0, 1.0 - complexity_diff)
        similarities.append(complexity_sim)
        
        return np.mean(similarities)

class LangChainCognitiveAgent(ABC):
    """Enhanced cognitive agent with LangChain integration"""
    
    def __init__(self, knowledge_graph: EnhancedKnowledgeGraph, agent_name: str):
        self.kg = knowledge_graph
        self.agent_name = agent_name
        self.llm = knowledge_graph.llm
        self.agent_knowledge = {}
        self.performance_history = []
        
        # Create agent-specific tools
        self.tools = self._create_agent_tools()
        
        # Create memory for conversations
        self.memory = MemorySaver()
        
        # Create LangGraph agent
        if self.llm and self.tools:
            self.langgraph_agent = create_react_agent(
                self.llm, self.tools, checkpointer=self.memory
            )
    
    def _create_agent_tools(self) -> List[Tool]:
        """Create tools specific to this agent"""
        return [
            Tool(
                name=f"{self.agent_name}_pattern_search",
                description=f"Search for cognitive patterns using {self.agent_name} expertise",
                func=self._pattern_search_tool
            ),
            Tool(
                name=f"{self.agent_name}_analyze_features",
                description=f"Analyze physiological features using {self.agent_name} methods",
                func=self._analyze_features_tool
            )
        ]
    
    def _pattern_search_tool(self, query: str) -> str:
        """Tool for searching relevant patterns"""
        # Mock implementation - in practice, parse query and search KG
        return f"{self.agent_name} found relevant patterns for: {query}"
    
    def _analyze_features_tool(self, features_summary: str) -> str:
        """Tool for analyzing physiological features"""
        return f"{self.agent_name} analysis: {features_summary}"
    
    @abstractmethod
    async def predict_with_knowledge(self, 
                                   features: np.ndarray,
                                   user_profile: UserProfile,
                                   task_context: TaskContext) -> Tuple[int, float, str]:
        """Predict cognitive load with reasoning"""
        pass

class LangGraphPhysiologicalAgent(LangChainCognitiveAgent):
    """Physiological pattern agent with LangGraph workflow"""
    
    def __init__(self, knowledge_graph: EnhancedKnowledgeGraph):
        super().__init__(knowledge_graph, "PhysiologicalAgent")
        
        # Create specialized workflow
        self.workflow = self._create_physiological_workflow()
    
    def _create_physiological_workflow(self) -> StateGraph:
        """Create LangGraph workflow for physiological analysis"""
        
        def analyze_features(state: AgentState) -> AgentState:
            """Analyze physiological features"""
            reasoning = [f"{self.agent_name}: Analyzing physiological features"]
            
            if state["features"] is not None:
                features = state["features"]
                avg_activation = np.mean(features)
                
                if avg_activation > 0.7:
                    load_indication = "high cognitive load"
                elif avg_activation > 0.4:
                    load_indication = "moderate cognitive load"
                else:
                    load_indication = "low cognitive load"
                
                reasoning.append(f"Feature analysis suggests {load_indication}")
            
            state["reasoning_chain"].extend(reasoning)
            return state
        
        def retrieve_patterns(state: AgentState) -> AgentState:
            """Retrieve similar physiological patterns"""
            reasoning = [f"{self.agent_name}: Retrieving similar patterns"]
            
            # Mock pattern retrieval
            if state["features"] is not None and state["user_profile"] and state["task_context"]:
                # In real implementation, use async pattern retrieval
                patterns = []  # Mock patterns
                state["retrieved_patterns"] = patterns
                reasoning.append(f"Found {len(patterns)} similar patterns")
            
            state["reasoning_chain"].extend(reasoning)
            return state
        
        def make_prediction(state: AgentState) -> AgentState:
            """Make final prediction based on analysis"""
            reasoning = [f"{self.agent_name}: Making prediction"]
            
            # Mock prediction logic
            prediction = {
                "class": 1,  # Moderate load
                "confidence": 0.75,
                "agent": self.agent_name
            }
            
            state["predictions"][self.agent_name] = prediction
            reasoning.append(f"Predicted class {prediction['class']} with confidence {prediction['confidence']}")
            
            state["reasoning_chain"].extend(reasoning)
            return state
        
        # Build workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze_features", analyze_features)
        workflow.add_node("retrieve_patterns", retrieve_patterns)
        workflow.add_node("make_prediction", make_prediction)
        
        workflow.add_edge("analyze_features", "retrieve_patterns")
        workflow.add_edge("retrieve_patterns", "make_prediction")
        workflow.add_edge("make_prediction", END)
        
        workflow.set_entry_point("analyze_features")
        
        return workflow.compile()
    
    async def predict_with_knowledge(self, 
                                   features: np.ndarray,
                                   user_profile: UserProfile,
                                   task_context: TaskContext) -> Tuple[int, float, str]:
        """Predict using LangGraph workflow"""
        
        initial_state = AgentState(
            messages=[],
            user_profile=user_profile,
            task_context=task_context,
            features=features,
            predictions={},
            retrieved_patterns=[],
            final_prediction=None,
            reasoning_chain=[]
        )
        
        # Run workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        # Extract prediction
        if self.agent_name in final_state["predictions"]:
            pred = final_state["predictions"][self.agent_name]
            reasoning = "; ".join(final_state["reasoning_chain"])
            return pred["class"], pred["confidence"], reasoning
        
        return 1, 0.5, "Default prediction"

class LangGraphTemporalAgent(LangChainCognitiveAgent):
    """Temporal pattern agent with sequence analysis"""
    
    def __init__(self, knowledge_graph: EnhancedKnowledgeGraph):
        super().__init__(knowledge_graph, "TemporalAgent")
        self.recent_history = []
        self.transition_matrix = np.ones((3, 3)) / 3
        
        self.workflow = self._create_temporal_workflow()
    
    def _create_temporal_workflow(self) -> StateGraph:
        """Create workflow for temporal pattern analysis"""
        
        def analyze_sequence(state: AgentState) -> AgentState:
            """Analyze temporal sequence patterns"""
            reasoning = [f"{self.agent_name}: Analyzing temporal patterns"]
            
            if len(self.recent_history) >= 2:
                last_state = self.recent_history[-1]['class']
                transition_probs = self.transition_matrix[last_state]
                predicted_class = np.argmax(transition_probs)
                confidence = transition_probs[predicted_class]
                
                reasoning.append(f"Based on transition from state {last_state}, predicting {predicted_class}")
                
                state["predictions"][self.agent_name] = {
                    "class": int(predicted_class),
                    "confidence": float(confidence),
                    "agent": self.agent_name
                }
            else:
                reasoning.append("Insufficient history for temporal analysis")
                state["predictions"][self.agent_name] = {
                    "class": 1,
                    "confidence": 0.33,
                    "agent": self.agent_name
                }
            
            state["reasoning_chain"].extend(reasoning)
            return state
        
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze_sequence", analyze_sequence)
        workflow.add_edge("analyze_sequence", END)
        workflow.set_entry_point("analyze_sequence")
        
        return workflow.compile()
    
    async def predict_with_knowledge(self, 
                                   features: np.ndarray,
                                   user_profile: UserProfile,
                                   task_context: TaskContext) -> Tuple[int, float, str]:
        """Predict using temporal workflow"""
        
        initial_state = AgentState(
            messages=[],
            user_profile=user_profile,
            task_context=task_context,
            features=features,
            predictions={},
            retrieved_patterns=[],
            final_prediction=None,
            reasoning_chain=[]
        )
        
        final_state = await self.workflow.ainvoke(initial_state)
        
        if self.agent_name in final_state["predictions"]:
            pred = final_state["predictions"][self.agent_name]
            reasoning = "; ".join(final_state["reasoning_chain"])
            return pred["class"], pred["confidence"], reasoning
        
        return 1, 0.33, "Default temporal prediction"

class LangGraphRAGCognitiveSystem:
    """Enhanced RAG system with LangGraph orchestration"""
    
    def __init__(self, base_model=None, llm_model=None):
        self.base_model = base_model or self._create_mock_model()
        self.llm = llm_model or init_chat_model("anthropic:claude-3-5-sonnet-latest")
        
        # Enhanced knowledge graph
        self.knowledge_graph = EnhancedKnowledgeGraph(self.llm)
        
        # Multi-agent system
        self.agents = {
            'physiological': LangGraphPhysiologicalAgent(self.knowledge_graph),
            'temporal': LangGraphTemporalAgent(self.knowledge_graph)
        }
        
        # Meta-learning weights
        self.meta_weights = {'physiological': 0.6, 'temporal': 0.4}
        
        # Main workflow
        self.main_workflow = self._create_main_workflow()
        
        # Memory for system state
        self.memory = MemorySaver()
        self.prediction_history = []
    
    def _create_mock_model(self):
        """Create mock base model"""
        class MockModel:
            def __init__(self):
                self.weights = np.random.rand(50) * 2 - 1
            
            def predict(self, features):
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                
                predictions = []
                for feature_vec in features:
                    min_len = min(len(feature_vec), len(self.weights))
                    score = np.dot(feature_vec[:min_len], self.weights[:min_len])
                    
                    if score < -0.5:
                        pred_class = 0
                    elif score > 0.5:
                        pred_class = 2
                    else:
                        pred_class = 1
                    
                    predictions.append(pred_class)
                
                return np.array(predictions)
        
        return MockModel()
    
    def _create_main_workflow(self) -> StateGraph:
        """Create main system workflow"""
        
        def base_prediction(state: AgentState) -> AgentState:
            """Get base model prediction"""
            if state["features"] is not None:
                base_pred = self.base_model.predict(state["features"].reshape(1, -1))[0]
                state["predictions"]["base"] = {
                    "class": int(base_pred),
                    "confidence": 0.5,
                    "agent": "base_model"
                }
                state["reasoning_chain"].append(f"Base model predicted class {base_pred}")
            return state
        
        async def agent_predictions(state: AgentState) -> AgentState:
            """Get predictions from all agents"""
            state["reasoning_chain"].append("Collecting agent predictions")
            
            # Run each agent's workflow
            for agent_name, agent in self.agents.items():
                try:
                    pred_class, confidence, reasoning = await agent.predict_with_knowledge(
                        state["features"], state["user_profile"], state["task_context"]
                    )
                    state["predictions"][agent_name] = {
                        "class": int(pred_class),
                        "confidence": float(confidence),
                        "agent": agent_name,
                        "reasoning": reasoning
                    }
                except Exception as e:
                    state["reasoning_chain"].append(f"Error in {agent_name}: {e}")
                    state["predictions"][agent_name] = {
                        "class": 1,
                        "confidence": 0.33,
                        "agent": agent_name,
                        "reasoning": f"Default due to error: {e}"
                    }
            
            return state
        
        def combine_predictions(state: AgentState) -> AgentState:
            """Combine all predictions using meta-learning"""
            predictions = state["predictions"]
            
            # Weighted voting
            class_votes = {0: 0.0, 1: 0.0, 2: 0.0}
            total_weight = 0.0
            
            # Add base model vote
            if "base" in predictions:
                base_weight = 0.3
                class_votes[predictions["base"]["class"]] += base_weight
                total_weight += base_weight
            
            # Add agent votes
            for agent_name, pred_info in predictions.items():
                if agent_name != "base":
                    agent_weight = self.meta_weights.get(agent_name, 0.3)
                    weighted_vote = agent_weight * pred_info["confidence"]
                    class_votes[pred_info["class"]] += weighted_vote
                    total_weight += weighted_vote
            
            # Final prediction
            if total_weight > 0:
                final_class = max(class_votes.keys(), key=lambda k: class_votes[k])
                confidence = min(class_votes[final_class] / total_weight, 0.95)
            else:
                final_class = 1
                confidence = 0.5
            
            state["final_prediction"] = {
                "class": final_class,
                "confidence": confidence,
                "class_votes": class_votes,
                "total_weight": total_weight
            }
            
            state["reasoning_chain"].append(
                f"Final prediction: class {final_class} with confidence {confidence:.3f}"
            )
            
            return state
        
        # Build main workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("base_prediction", base_prediction)
        workflow.add_node("agent_predictions", agent_predictions)
        workflow.add_node("combine_predictions", combine_predictions)
        
        workflow.add_edge("base_prediction", "agent_predictions")
        workflow.add_edge("agent_predictions", "combine_predictions")
        workflow.add_edge("combine_predictions", END)
        
        workflow.set_entry_point("base_prediction")
        
        return workflow.compile()
    
    async def predict_with_rag_async(self, 
                                   features: np.ndarray,
                                   user_profile: UserProfile,
                                   task_context: TaskContext) -> Dict[str, Any]:
        """Enhanced async prediction using LangGraph workflow"""
        
        # Initialize state
        initial_state = AgentState(
            messages=[],
            user_profile=user_profile,
            task_context=task_context,
            features=features,
            predictions={},
            retrieved_patterns=[],
            final_prediction=None,
            reasoning_chain=[]
        )
        
        # Run main workflow
        final_state = await self.main_workflow.ainvoke(initial_state)
        
        # Retrieve supporting evidence
        supporting_patterns = await self.knowledge_graph.retrieve_similar_patterns_semantic(
            features, user_profile, task_context
        )
        
        # Build result
        result = {
            'prediction': final_state["final_prediction"]["class"],
            'confidence': final_state["final_prediction"]["confidence"],
            'predictions_breakdown': final_state["predictions"],
            'supporting_evidence': len(supporting_patterns),
            'reasoning_chain': final_state["reasoning_chain"],
            'explanation': self._generate_explanation(
                final_state["final_prediction"], 
                supporting_patterns, 
                final_state["predictions"]
            )
        }
        
        # Store prediction for analysis
        self.prediction_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'user_id': user_profile.user_id,
            'prediction': result['prediction'],
            'confidence': result['confidence']
        })
        
        return result
    
    def predict_with_rag(self, features: np.ndarray,
                        user_profile: UserProfile,
                        task_context: TaskContext) -> Dict[str, Any]:
        """Synchronous wrapper for async prediction"""
        return asyncio.run(self.predict_with_rag_async(features, user_profile, task_context))
    
    async def update_from_feedback_async(self, 
                                       features: np.ndarray,
                                       user_profile: UserProfile,
                                       task_context: TaskContext,
                                       ground_truth: int):
        """Enhanced feedback update with reasoning generation"""
        
        # Generate reasoning for this pattern
        reasoning = await self._generate_feedback_reasoning(
            features, user_profile, task_context, ground_truth
        )
        
        # Create new pattern
        pattern = CognitivePattern(
            pattern_id=f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            features=features,
            cognitive_load_class=ground_truth,
            user_context=user_profile,
            task_context=task_context,
            confidence=1.0,
            reasoning=reasoning
        )
        
        # Add to knowledge graph with reasoning
        await self.knowledge_graph.add_pattern_with_reasoning(pattern, reasoning)
        
        # Update agent knowledge
        for agent in self.agents.values():
            try:
                agent.update_knowledge(pattern, ground_truth)
            except Exception as e:
                print(f"Error updating agent knowledge: {e}")
        
        print(f"Knowledge updated with ground truth: {ground_truth}")
        print(f"Reasoning: {reasoning}")
    
    async def _generate_feedback_reasoning(self, 
                                         features: np.ndarray,
                                         user_profile: UserProfile,
                                         task_context: TaskContext,
                                         ground_truth: int) -> str:
        """Generate reasoning for feedback pattern using LLM"""
        
        prompt = ChatPromptTemplate.from_template(
            """Based on the feedback provided, explain why this combination leads to {cognitive_load} cognitive load:
            
            User: {expertise} level, Traits: {traits}
            Task: {task_type} with complexity {complexity}
            Result: {cognitive_load_name} cognitive load
            
            Provide insights on what factors contributed to this cognitive load level."""
        )
        
        class_names = {0: "Low", 1: "Moderate", 2: "High"}
        
        messages = prompt.format_messages(
            cognitive_load=class_names[ground_truth],
            expertise=user_profile.expertise_level,
            traits=str(user_profile.cognitive_traits),
            task_type=task_context.task_type,
            complexity=task_context.complexity_level,
            cognitive_load_name=class_names[ground_truth]
        )
        
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def _generate_explanation(self, prediction: Dict, 
                            supporting_patterns: List,
                            agent_predictions: Dict) -> str:
        """Generate comprehensive explanation"""
        class_names = {0: "Low", 1: "Moderate", 2: "High"}
        pred_class_name = class_names.get(prediction['class'], "Unknown")
        
        explanation = f"Predicted {pred_class_name} cognitive load (class {prediction['class']}) with {prediction['confidence']:.3f} confidence. "
        
        if supporting_patterns:
            explanation += f"Supported by {len(supporting_patterns)} similar patterns. "
        
        # Add agent reasoning
        agent_reasoning = []
        for agent_name, pred_info in agent_predictions.items():
            if agent_name != "base" and "reasoning" in pred_info:
                agent_reasoning.append(f"{agent_name}: {pred_info['reasoning']}")
        
        if agent_reasoning:
            explanation += f"Agent analysis: {'; '.join(agent_reasoning)}"
        
        return explanation

# Demo function with LangChain integration
async def run_enhanced_demo():
    """Run demonstration of enhanced RAG system"""
    print("🧠 Enhanced RAG-Cognitive Load System with LangChain Integration")
    print("=" * 70)
    
    # Initialize enhanced system
    rag_system = LangGraphRAGCognitiveSystem()
    
    # Create sample users and tasks
    users = [
        UserProfile(
            user_id="user_001",
            demographics={"age": 25, "education_years": 16, "gender": "F"},
            cognitive_traits={"working_memory": 0.8, "attention_span": 0.7, "processing_speed": 0.6},
            historical_performance={"accuracy": 0.85, "reaction_time": 0.6},
            expertise_level="intermediate"
        ),
        UserProfile(
            user_id="user_002", 
            demographics={"age": 30, "education_years": 18, "gender": "M"},
            cognitive_traits={"working_memory": 0.9, "attention_span": 0.8, "processing_speed": 0.7},
            historical_performance={"accuracy": 0.90, "reaction_time": 0.5},
            expertise_level="advanced"
        )
    ]
    
    tasks = [
        TaskContext("cognitive_test", "mathematics", 0.7, 300.0),
        TaskContext("memory_task", "verbal", 0.5, 180.0),
        TaskContext("attention_task", "visual", 0.8, 240.0)
    ]
    
    print("\n1. Building Enhanced Knowledge Base...")
    
    # Build knowledge base with reasoning
    for i in range(10):  # Reduced for demo
        user = random.choice(users)
        task = random.choice(tasks)
        features = np.random.rand(50)
        
        # Simulate realistic cognitive load
        complexity_factor = task.complexity_level
        expertise_factor = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.9}[user.expertise_level]
        load_score = complexity_factor - expertise_factor + np.random.normal(0, 0.2)
        
        if load_score < 0.2:
            true_class = 0
        elif load_score < 0.6:
            true_class = 1
        else:
            true_class = 2
        
        features += true_class * 0.1 * np.random.rand(50)
        
        # Update with async feedback
        await rag_system.update_from_feedback_async(features, user, task, true_class)
    
    print(f"✅ Added {len(rag_system.knowledge_graph._get_all_patterns())} patterns with reasoning")
    
    print("\n2. Making Enhanced Predictions...")
    
    # Test prediction with full workflow
    test_user = users[0]
    test_task = tasks[0]
    test_features = np.random.rand(50) + 0.5
    
    result = await rag_system.predict_with_rag_async(test_features, test_user, test_task)
    
    print(f"\n📊 Enhanced Prediction Results:")
    print(f"   Final Prediction: {result['prediction']} ({['Low', 'Moderate', 'High'][result['prediction']]})")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Supporting Evidence: {result['supporting_evidence']} patterns")
    
    print(f"\n🤖 Agent Breakdown:")
    for agent_name, pred_info in result['predictions_breakdown'].items():
        class_name = ['Low', 'Moderate', 'High'][pred_info['class']]
        print(f"   {agent_name}: {class_name} (conf: {pred_info['confidence']:.3f})")
        if 'reasoning' in pred_info:
            print(f"      Reasoning: {pred_info['reasoning'][:100]}...")
    
    print(f"\n🔍 Reasoning Chain:")
    for i, step in enumerate(result['reasoning_chain'], 1):
        print(f"   {i}. {step}")
    
    print(f"\n📝 Explanation: {result['explanation']}")
    
    print("\n✨ Enhanced demo completed!")
    print("\nNew LangChain/LangGraph Features:")
    print("• Multi-agent LangGraph workflows")
    print("• Semantic pattern retrieval")
    print("• LLM-generated reasoning chains")
    print("• Enhanced memory management")
    print("• Async processing capabilities")
    
    return rag_system

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Enhanced RAG Cognitive Load System...")
    system = asyncio.run(run_enhanced_demo())
    print("\n🔧 Enhanced system ready for deployment!")a
