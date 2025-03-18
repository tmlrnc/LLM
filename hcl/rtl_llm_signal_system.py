import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from enum import Enum
import time
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import uuid

# Hypothetical imports - would need actual implementations
from vector_db import VectorDatabase
from llm_api_wrapper import LLMModel, LLMMessage, ModelType
from rtl_validator import RTLValidator, ValidationLevel, HDLLanguage
from eda_tools import SimulationRunner, SynthesisRunner
from feedback_db import FeedbackDatabase

class AgentRole(Enum):
    """Defines specialized roles for RTL design agents"""
    REQUIREMENTS_ANALYZER = "requirements_analyzer"
    PATTERN_MATCHER = "pattern_matcher"
    RTL_GENERATOR = "rtl_generator"
    TESTBENCH_CREATOR = "testbench_creator"
    VERIFICATION_AGENT = "verification_agent"
    OPTIMIZER_AGENT = "optimizer_agent"
    DOCUMENTATION_AGENT = "documentation_agent"
    ORCHESTRATOR = "orchestrator"

@dataclass
class AgentConfig:
    """Configuration for an individual agent"""
    role: AgentRole
    model_type: ModelType
    temperature: float = 0.2
    top_p: float = 0.95
    prompt_template: str = ""
    max_tokens: int = 4000
    example_count: int = 3

@dataclass
class RTLCandidate:
    """Represents a candidate RTL design implementation"""
    id: str
    code: str
    language: HDLLanguage
    module_name: str
    description: str
    generator_agent_id: str
    generated_at: datetime
    scores: Dict[str, float] = None
    verification_results: Dict[str, Any] = None
    synthesis_results: Dict[str, Any] = None
    sme_feedback: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.scores is None:
            self.scores = {}
        if self.verification_results is None:
            self.verification_results = {}
        if self.synthesis_results is None:
            self.synthesis_results = {}
        if self.sme_feedback is None:
            self.sme_feedback = []

@dataclass
class DesignRequirement:
    """Represents parsed design requirements"""
    id: str
    raw_text: str
    parsed_components: Dict[str, Any]
    constraints: Dict[str, Any]
    identified_patterns: List[str] = None
    
    def __post_init__(self):
        if self.identified_patterns is None:
            self.identified_patterns = []

class RTLAgentSystem:
    """
    Main class for orchestrating RTL design agents, re-ranking,
    and integrating expert feedback.
    """
    
    def __init__(self, config_path: str):
        """Initialize the RTL agent system with configuration"""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.llm_client = LLMModel(api_key=self.config['api_key'])
        self.vector_db = VectorDatabase(self.config['vector_db_path'])
        self.rtl_validator = RTLValidator()
        self.feedback_db = FeedbackDatabase(self.config['feedback_db_path'])
        
        # Initialize agent configs
        self.agents = self._initialize_agents()
        
        # For tracking design projects
        self.active_designs = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _initialize_agents(self) -> Dict[AgentRole, AgentConfig]:
        """Initialize agent configurations for each role"""
        agents = {}
        for agent_config in self.config['agents']:
            role = AgentRole(agent_config['role'])
            agents[role] = AgentConfig(
                role=role,
                model_type=ModelType(agent_config['model_type']),
                temperature=agent_config.get('temperature', 0.2),
                top_p=agent_config.get('top_p', 0.95),
                prompt_template=self._load_prompt_template(agent_config['prompt_template_path']),
                max_tokens=agent_config.get('max_tokens', 4000),
                example_count=agent_config.get('example_count', 3)
            )
        return agents
    
    def _load_prompt_template(self, template_path: str) -> str:
        """Load prompt template from file"""
        full_path = os.path.join(self.config['templates_path'], template_path)
        with open(full_path, 'r') as f:
            return f.read()
    
    def create_design_project(self, project_name: str, requirements_text: str) -> str:
        """
        Create a new RTL design project based on requirements
        
        Args:
            project_name: Name of the design project
            requirements_text: Natural language requirements description
            
        Returns:
            design_id: Identifier for the new design project
        """
        design_id = str(uuid.uuid4())
        
        # Parse requirements using the requirements analyzer agent
        parsed_requirements = self._analyze_requirements(requirements_text)
        
        # Identify applicable design patterns using the pattern matcher agent
        patterns = self._match_design_patterns(parsed_requirements)
        parsed_requirements.identified_patterns = patterns
        
        # Store in active designs
        self.active_designs[design_id] = {
            'name': project_name,
            'requirements': parsed_requirements,
            'candidates': [],
            'selected_candidate_id': None,
            'testbenches': [],
            'verification_results': {},
            'created_at': datetime.now(),
            'status': 'REQUIREMENTS_PARSED'
        }
        
        self.logger.info(f"Created design project {project_name} with ID {design_id}")
        return design_id
    
    def _analyze_requirements(self, requirements_text: str) -> DesignRequirement:
        """
        Analyze requirements using the requirements analyzer agent
        
        Args:
            requirements_text: Natural language requirements description
            
        Returns:
            Structured requirements information
        """
        agent_config = self.agents[AgentRole.REQUIREMENTS_ANALYZER]
        
        # Prepare prompt with requirements text
        prompt = agent_config.prompt_template.format(requirements=requirements_text)
        
        # Retrieve similar requirements for few-shot examples
        similar_requirements = self.vector_db.search_similar(
            requirements_text, 
            collection="requirements",
            limit=agent_config.example_count
        )
        
        # Add examples to the prompt
        if similar_requirements:
            examples = "\n\n".join([
                f"Example {i+1}:\nRequirements: {ex['input']}\nParsed: {json.dumps(ex['output'], indent=2)}"
                for i, ex in enumerate(similar_requirements)
            ])
            prompt += f"\n\nHere are some examples of requirements parsing:\n{examples}"
        
        # Get response from LLM
        messages = [LLMMessage(role="system", content=prompt)]
        response = self.llm_client.generate(
            messages=messages,
            temperature=agent_config.temperature,
            top_p=agent_config.top_p,
            max_tokens=agent_config.max_tokens,
            model=agent_config.model_type
        )
        
        # Parse the structured output from the response
        try:
            # Extract JSON from response
            json_str = self._extract_json(response.content)
            parsed_data = json.loads(json_str)
            
            # Create requirement object
            requirement = DesignRequirement(
                id=str(uuid.uuid4()),
                raw_text=requirements_text,
                parsed_components=parsed_data.get('components', {}),
                constraints=parsed_data.get('constraints', {})
            )
            
            # Store the requirement for future reference
            self.vector_db.add_item(
                collection="requirements",
                item={
                    "input": requirements_text,
                    "output": parsed_data,
                    "id": requirement.id
                }
            )
            
            return requirement
            
        except Exception as e:
            self.logger.error(f"Failed to parse requirements: {str(e)}")
            # Return a basic structure if parsing fails
            return DesignRequirement(
                id=str(uuid.uuid4()),
                raw_text=requirements_text,
                parsed_components={},
                constraints={}
            )
    
    def _match_design_patterns(self, requirements: DesignRequirement) -> List[str]:
        """
        Match requirements to known design patterns using the pattern matcher agent
        
        Args:
            requirements: Structured requirements information
            
        Returns:
            List of identified design patterns
        """
        agent_config = self.agents[AgentRole.PATTERN_MATCHER]
        
        # Prepare prompt with requirements
        prompt = agent_config.prompt_template.format(
            raw_requirements=requirements.raw_text,
            parsed_requirements=json.dumps(requirements.parsed_components, indent=2),
            constraints=json.dumps(requirements.constraints, indent=2)
        )
        
        # Get response from LLM
        messages = [LLMMessage(role="system", content=prompt)]
        response = self.llm_client.generate(
            messages=messages,
            temperature=agent_config.temperature,
            top_p=agent_config.top_p,
            max_tokens=agent_config.max_tokens,
            model=agent_config.model_type
        )
        
        # Parse the pattern list from the response
        try:
            # Extract JSON from response
            json_str = self._extract_json(response.content)
            patterns_data = json.loads(json_str)
            
            # Return the list of patterns
            return patterns_data.get('patterns', [])
            
        except Exception as e:
            self.logger.error(f"Failed to match design patterns: {str(e)}")
            return []
    
    def generate_rtl_candidates(self, design_id: str, count: int = 3) -> List[str]:
        """
        Generate multiple RTL implementations based on requirements
        
        Args:
            design_id: ID of the design project
            count: Number of candidates to generate
            
        Returns:
            List of candidate IDs
        """
        if design_id not in self.active_designs:
            raise ValueError(f"Design with ID {design_id} not found")
        
        design = self.active_designs[design_id]
        requirements = design['requirements']
        
        # Generate multiple candidates
        candidate_ids = []
        for i in range(count):
            try:
                # Generate candidate with slightly different temperature for variation
                temperature_variation = 0.1 * (i / count)
                candidate = self._generate_rtl_candidate(
                    requirements, 
                    temperature_boost=temperature_variation
                )
                
                # Store candidate
                design['candidates'].append(candidate)
                candidate_ids.append(candidate.id)
                
            except Exception as e:
                self.logger.error(f"Failed to generate candidate {i}: {str(e)}")
        
        if candidate_ids:
            design['status'] = 'CANDIDATES_GENERATED'
            
        return candidate_ids
    
    def _generate_rtl_candidate(self, 
                               requirements: DesignRequirement, 
                               temperature_boost: float = 0.0) -> RTLCandidate:
        """
        Generate a single RTL candidate implementation
        
        Args:
            requirements: Structured requirements information
            temperature_boost: Optional increase to temperature for variation
            
        Returns:
            Generated RTL candidate
        """
        agent_config = self.agents[AgentRole.RTL_GENERATOR]
        
        # Determine language to use (could be based on requirements or config)
        language = HDLLanguage.VERILOG  # Default
        if 'preferred_language' in requirements.constraints:
            lang_str = requirements.constraints['preferred_language'].lower()
            if 'vhdl' in lang_str:
                language = HDLLanguage.VHDL
            elif 'systemverilog' in lang_str:
                language = HDLLanguage.SYSTEMVERILOG
        
        # Retrieve relevant examples from vector DB based on patterns
        examples = []
        for pattern in requirements.identified_patterns:
            pattern_examples = self.vector_db.search_similar(
                pattern,
                collection="rtl_patterns",
                limit=1
            )
            examples.extend(pattern_examples)
        
        # If we don't have enough pattern-specific examples, get general ones
        if len(examples) < agent_config.example_count:
            general_examples = self.vector_db.search_similar(
                requirements.raw_text,
                collection="rtl_implementations",
                limit=agent_config.example_count - len(examples)
            )
            examples.extend(general_examples)
        
        # Prepare prompt with requirements and examples
        prompt = agent_config.prompt_template.format(
            raw_requirements=requirements.raw_text,
            parsed_requirements=json.dumps(requirements.parsed_components, indent=2),
            constraints=json.dumps(requirements.constraints, indent=2),
            identified_patterns=", ".join(requirements.identified_patterns),
            target_language=language.value
        )
        
        # Add examples to the prompt
        if examples:
            examples_text = "\n\n".join([
                f"Example {i+1} ({ex['pattern']}):\n```{ex['language']}\n{ex['implementation']}\n```"
                for i, ex in enumerate(examples)
            ])
            prompt += f"\n\nHere are some relevant examples:\n{examples_text}"
        
        # Get response from LLM
        messages = [LLMMessage(role="system", content=prompt)]
        response = self.llm_client.generate(
            messages=messages,
            temperature=agent_config.temperature + temperature_boost,
            top_p=agent_config.top_p,
            max_tokens=agent_config.max_tokens,
            model=agent_config.model_type
        )
        
        # Extract code and metadata from response
        code_block = self._extract_code_block(response.content)
        
        # Try to extract module/entity name from the code
        module_name = self._extract_module_name(code_block, language)
        
        # Extract description if provided
        description = self._extract_description(response.content) or "Generated RTL implementation"
        
        # Create candidate object
        candidate = RTLCandidate(
            id=str(uuid.uuid4()),
            code=code_block,
            language=language,
            module_name=module_name,
            description=description,
            generator_agent_id=str(agent_config.role.value),
            generated_at=datetime.now()
        )
        
        return candidate
    
    def evaluate_rtl_candidates(self, design_id: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate and rank multiple RTL candidates using various metrics
        
        Args:
            design_id: ID of the design project
            
        Returns:
            Dictionary mapping candidate IDs to their scores
        """
        if design_id not in self.active_designs:
            raise ValueError(f"Design with ID {design_id} not found")
        
        design = self.active_designs[design_id]
        
        # Evaluate each candidate
        candidate_scores = {}
        for candidate in design['candidates']:
            scores = self._evaluate_candidate(candidate, design['requirements'])
            candidate.scores = scores
            candidate_scores[candidate.id] = scores
            
        # Store the updated candidates
        self.active_designs[design_id]['status'] = 'CANDIDATES_EVALUATED'
        
        return candidate_scores
    
    def _evaluate_candidate(self, 
                           candidate: RTLCandidate, 
                           requirements: DesignRequirement) -> Dict[str, float]:
        """
        Evaluate a single RTL candidate using multiple metrics
        
        Args:
            candidate: RTL candidate to evaluate
            requirements: Requirements to evaluate against
            
        Returns:
            Dictionary of evaluation scores
        """
        scores = {}
        
        # Syntax correctness (0-100)
        syntax_valid, syntax_errors = self.rtl_validator.validate_rtl(
            candidate.code, 
            candidate.language, 
            ValidationLevel.SYNTAX
        )
        scores['syntax'] = 100.0 if syntax_valid else max(0, 100 - (len(syntax_errors) * 10))
        
        # Linting quality (0-100)
        lint_valid, lint_warnings = self.rtl_validator.validate_rtl(
            candidate.code, 
            candidate.language, 
            ValidationLevel.LINT
        )
        # A handful of lint warnings is ok, but many indicate poor quality
        warning_count = len(lint_warnings)
        scores['lint_quality'] = max(0, 100 - (warning_count * 5))
        
        # Synthesizability (0-100)
        synth_valid, synth_errors = self.rtl_validator.validate_rtl(
            candidate.code, 
            candidate.language, 
            ValidationLevel.SYNTHESIZABILITY,
            candidate.module_name
        )
        scores['synthesizability'] = 100.0 if synth_valid else max(0, 100 - (len(synth_errors) * 10))
        
        # Code complexity (0-100, higher is better/simpler)
        complexity_score = self._calculate_complexity_score(candidate.code, candidate.language)
        scores['simplicity'] = complexity_score
        
        # Requirements coverage (0-100)
        coverage_score = self._calculate_requirements_coverage(candidate.code, requirements)
        scores['requirements_coverage'] = coverage_score
        
        # Calculate weighted average score (0-100)
        weights = {
            'syntax': 0.2,
            'lint_quality': 0.15,
            'synthesizability': 0.25,
            'simplicity': 0.15,
            'requirements_coverage': 0.25
        }
        
        weighted_sum = sum(scores[key] * weights[key] for key in weights)
        scores['overall'] = weighted_sum
        
        return scores
    
    def _calculate_complexity_score(self, code: str, language: HDLLanguage) -> float:
        """Calculate code complexity score (higher is better/simpler)"""
        # Count lines, statements, nested blocks, etc.
        line_count = len(code.strip().split('\n'))
        
        # Excessive length penalizes the score
        if line_count > 500:
            length_factor = 0.7
        elif line_count > 300:
            length_factor = 0.8
        elif line_count > 150:
            length_factor = 0.9
        else:
            length_factor = 1.0
            
        # Count nested blocks
        nested_level = 0
        max_nested = 0
        for line in code.split('\n'):
            # Different patterns for different languages
            if language == HDLLanguage.VHDL:
                if any(keyword in line.lower() for keyword in ['begin', 'then', 'loop']):
                    nested_level += 1
                if any(keyword in line.lower() for keyword in ['end', 'end if', 'end loop']):
                    nested_level -= 1
            else:  # Verilog/SystemVerilog
                if 'begin' in line:
                    nested_level += 1
                if 'end' in line:
                    nested_level -= 1
            
            max_nested = max(max_nested, nested_level)
        
        # Penalize for deep nesting
        if max_nested > 5:
            nesting_factor = 0.7
        elif max_nested > 3:
            nesting_factor = 0.85
        else:
            nesting_factor = 1.0
            
        # Base score calculation - could be refined with more metrics
        base_score = 90.0  # Start with assumption of good code
        
        # Apply modifiers
        final_score = base_score * length_factor * nesting_factor
        
        return min(100.0, final_score)
    
    def _calculate_requirements_coverage(self, code: str, requirements: DesignRequirement) -> float:
        """Calculate how well the code covers the requirements"""
        # This is a simplified approach - real implementation would need more sophisticated analysis
        
        # Extract key terms from requirements
        key_terms = []
        for component in requirements.parsed_components.values():
            if isinstance(component, dict):
                key_terms.extend(component.keys())
            elif isinstance(component, str):
                key_terms.append(component)
                
        for constraint in requirements.constraints.values():
            if isinstance(constraint, str):
                key_terms.append(constraint)
        
        # Clean up terms
        key_terms = [term.lower() for term in key_terms if len(term) > 3]
        key_terms = list(set(key_terms))  # Remove duplicates
        
        # Count how many key terms appear in the code
        code_lower = code.lower()
        found_terms = sum(1 for term in key_terms if term in code_lower)
        
        # Calculate coverage percentage
        if not key_terms:
            return 90.0  # Default if no clear terms
            
        coverage_pct = (found_terms / len(key_terms)) * 100
        
        # Apply a curve to reward high coverage
        if coverage_pct >= 90:
            return 100.0
        elif coverage_pct >= 75:
            return 90.0
        elif coverage_pct >= 60:
            return 80.0
        elif coverage_pct >= 50:
            return 70.0
        else:
            return max(50.0, coverage_pct)  # Minimum score of 50
    
    def rank_candidates(self, design_id: str) -> List[str]:
        """
        Rank RTL candidates based on their evaluation scores
        
        Args:
            design_id: ID of the design project
            
        Returns:
            List of candidate IDs in ranked order (best first)
        """
        if design_id not in self.active_designs:
            raise ValueError(f"Design with ID {design_id} not found")
        
        design = self.active_designs[design_id]
        
        # Sort candidates by overall score
        ranked_candidates = sorted(
            design['candidates'], 
            key=lambda c: c.scores.get('overall', 0) if c.scores else 0,
            reverse=True
        )
        
        # Update design with ranked candidate IDs
        ranked_ids = [c.id for c in ranked_candidates]
        
        # Auto-select the best candidate
        if ranked_ids:
            design['selected_candidate_id'] = ranked_ids[0]
            design['status'] = 'CANDIDATE_SELECTED'
        
        return ranked_ids
    
    def generate_testbench(self, design_id: str, candidate_id: Optional[str] = None) -> str:
        """
        Generate a testbench for a specific RTL candidate
        
        Args:
            design_id: ID of the design project
            candidate_id: ID of the specific candidate (uses selected candidate if None)
            
        Returns:
            Testbench ID
        """
        if design_id not in self.active_designs:
            raise ValueError(f"Design with ID {design_id} not found")
        
        design = self.active_designs[design_id]
        
        # Use selected candidate if none specified
        if candidate_id is None:
            candidate_id = design['selected_candidate_id']
            if candidate_id is None:
                raise ValueError("No candidate selected for this design")
        
        # Find the candidate
        candidate = next((c for c in design['candidates'] if c.id == candidate_id), None)
        if candidate is None:
            raise ValueError(f"Candidate with ID {candidate_id} not found")
        
        # Generate testbench using the testbench creator agent
        testbench = self._create_testbench(candidate, design['requirements'])
        
        # Store the testbench
        design['testbenches'].append(testbench)
        design['status'] = 'TESTBENCH_GENERATED'
        
        return testbench['id']
    
    def _create_testbench(self, 
                         candidate: RTLCandidate, 
                         requirements: DesignRequirement) -> Dict[str, Any]:
        """
        Create a testbench for a specific RTL candidate
        
        Args:
            candidate: RTL candidate to test
            requirements: Design requirements
            
        Returns:
            Dictionary containing testbench information
        """
        agent_config = self.agents[AgentRole.TESTBENCH_CREATOR]
        
        # Prepare prompt with candidate code and requirements
        prompt = agent_config.prompt_template.format(
            rtl_code=candidate.code,
            module_name=candidate.module_name,
            language=candidate.language.value,
            raw_requirements=requirements.raw_text,
            constraints=json.dumps(requirements.constraints, indent=2)
        )
        
        # Get response from LLM
        messages = [LLMMessage(role="system", content=prompt)]
        response = self.llm_client.generate(
            messages=messages,
            temperature=agent_config.temperature,
            top_p=agent_config.top_p,
            max_tokens=agent_config.max_tokens,
            model=agent_config.model_type
        )
        
        # Extract code and metadata from response
        testbench_code = self._extract_code_block(response.content)
        
        # Create testbench dictionary
        testbench = {
            'id': str(uuid.uuid4()),
            'candidate_id': candidate.id,
            'code': testbench_code,
            'language': candidate.language.value,
            'created_at': datetime.now()
        }
        
        return testbench
    
    def run_verification(self, design_id: str, testbench_id: str) -> Dict[str, Any]:
        """
        Run verification using a generated testbench
        
        Args:
            design_id: ID of the design project
            testbench_id: ID of the testbench to use
            
        Returns:
            Verification results
        """
        if design_id not in self.active_designs:
            raise ValueError(f"Design with ID {design_id} not found")
        
        design = self.active_designs[design_id]
        
        # Find the testbench
        testbench = next((tb for tb in design['testbenches'] if tb['id'] == testbench_id), None)
        if testbench is None:
            raise ValueError(f"Testbench with ID {testbench_id} not found")
        
        # Find the associated candidate
        candidate_id = testbench['candidate_id']
        candidate = next((c for c in design['candidates'] if c.id == candidate_id), None)
        if candidate is None:
            raise ValueError(f"Candidate with ID {candidate_id} not found")
        
        # Initialize simulation runner
        simulator = SimulationRunner()
        
        # Run simulation
        sim_results = simulator.run_simulation(
            rtl_code=candidate.code,
            testbench_code=testbench['code'],
            language=HDLLanguage(testbench['language'])
        )
        
        # Store verification results
        verification_result = {
            'id': str(uuid.uuid4()),
            'testbench_id': testbench_id,
            'candidate_id': candidate_id,
            'simulation_results': sim_results,
            'run_at': datetime.now(),
            'status': 'SUCCESS' if sim_results['pass'] else 'FAILURE',
            'analysis': None
        }
        
        # Analyze failures if any
        if not sim_results['pass']:
            verification_result['analysis'] = self._analyze_verification_failures(
                candidate, testbench, sim_results
            )
        
        # Store results
        design['verification_results'][verification_result['id']] = verification_result
        design['status'] = 'VERIFICATION_COMPLETE'
        
        # Update candidate verification results
        candidate.verification_results = sim_results
        
        return verification_result
    
    def _analyze_verification_failures(self, 
                                      candidate: RTLCandidate, 
                                      testbench: Dict[str, Any], 
                                      sim_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze verification failures using the verification agent
        
        Args:
            candidate: RTL candidate
            testbench: Testbench that was used
            sim_results: Simulation results
            
        Returns:
            Analysis dictionary
        """
        agent_config = self.agents[AgentRole.VERIFICATION_AGENT]
        
        # Prepare prompt with failure information
        prompt = agent_config.prompt_template.format(
            rtl_code=candidate.code,
            testbench_code=testbench['code'],
            simulation_output=json.dumps(sim_results, indent=2),
            errors="\n".join(sim_results.get('errors', [])),
            warnings="\n".join(sim_results.get('warnings', []))
        )
        
        # Get response from LLM
        messages = [LLMMessage(role="system", content=prompt)]
        response = self.llm_client.generate(
            messages=messages,
            temperature=agent_config.temperature,
            top_p=agent_config.top_p,
            max_tokens=agent_config.max_tokens,
            model=agent_config.model_type
        )
        
        # Extract analysis from response
        try:
            analysis_json = self._extract_json(response.content)
            analysis = json.loads(analysis_json)
            return analysis
        except Exception as e:
            self.logger.error(f"Failed to parse verification analysis: {str(e)}")
            return {
                "error": "Failed to parse analysis",
                "raw_response": response.content
            }
    
    def collect_sme_feedback(self, design_id: str, candidate_id: str, feedback: Dict[str, Any]) -> str:
        """
        Collect and store SME (Subject Matter Expert) feedback on an RTL candidate
        
        Args:
            design_id: ID of the design project
            candidate_id: ID of the candidate
            feedback: Dictionary containing feedback information
            
        Returns:
            Feedback ID
        """
        if design_id not in self.active_designs:
            raise ValueError(f"Design with ID {design_id} not found")
        
        design = self.active_designs[design_id]
        
        # Find the candidate
        candidate = next((c for c in design['candidates'] if c.id == candidate_id), None)
        if candidate is None:
            raise ValueError(f"Candidate with ID {candidate_id} not found")
        
        # Create feedback record
        feedback_id = str(uuid.uuid4())
        feedback_record = {
            'id': feedback_id,
            'design_id': design_id,
            'candidate_id': candidate_id,
            'feedback': feedback,
            'submitted_at': datetime.now(),
            'status': 'NEW'
        }
        
        # Store in database
        self.feedback_db.add_feedback(feedback_record)
        
        # Add to candidate
        candidate.sme_feedback.append(feedback)
        
        # Update design status
        design['status'] = 'FEEDBACK_RECEIVED'
        
        # Schedule feedback processing for agent improvement
        self._schedule_feedback_processing(feedback_record)
        
        return feedback_id
    
    def _schedule_feedback_processing(self, feedback_record: Dict[str, Any]) -> None:
        """
        Schedule feedback processing for agent improvement
        
        Args:
            feedback_record: Feedback information to process
        """
        # In a real implementation, this might use a task queue or background worker
        # For this example, we'll just process it immediately
        self._process_feedback(feedback_record)
    
    def _process_feedback(self, feedback_record: Dict[str, Any]) -> None:
        """
        Process SME feedback to improve agents
        
        Args:
            feedback_record: Feedback information to process
        """
        try:
            # Find relevant data
            design_id = feedback_record['design_id']
            candidate_id = feedback_record['candidate_id']
            feedback = feedback_record['feedback']
            
            if design_id not in self.active_designs:
                self.logger.error(f"Design with ID {design_id} not found for feedback processing")
                return
                
            design = self.active_designs[design_id]
            
            # Find the candidate
            candidate = next((c for c in design['candidates'] if c.id == candidate_id), None)
            if candidate is None:
                self.logger.error(f"Candidate with ID {candidate_id} not found for feedback processing")
                return
            
            # Identify which agent needs improvement
            agent_role = AgentRole(candidate.generator_agent_id)
            
            # Store feedback for future fine-tuning
            self.feedback_db.add_training_example(
                agent_role=agent_role.value,
                input_data={
                    'requirements': design['requirements'].raw_text,
                    'parsed_requirements': design['requirements'].parsed_components,
                    'constraints': design['requirements'].constraints,
                    'patterns': design['requirements'].identified_patterns
                },
                expected_output=feedback.get('corrected_code', candidate.code),
                feedback_summary=feedback.get('summary', ''),
                feedback_rating=feedback.get('rating', 0)
            )
            
            # Mark feedback as processed
            self.feedback_db.update_feedback_status(feedback_record['id'], 'PROCESSED')
            
            self.logger.info(f"Processed feedback {feedback_record['id']} for agent improvement")
            
        except Exception as e:
            self.logger.error(f"Error processing feedback: {str(e)}")
    
    def improve_rtl_with_feedback(self, design_id: str, candidate_id: str, feedback_id: str) -> str:
        """
        Generate improved RTL based on SME feedback
        
        Args:
            design_id: ID of the design project
            candidate_id: ID of the candidate to improve
            feedback_id: ID of the feedback to use
            
        Returns:
            New candidate ID
        """
        if design_id not in self.active_designs:
            raise ValueError(f"Design with ID {design_id} not found")
        
        design = self.active_designs[design_id]
        
        # Find the candidate
        candidate = next((c for c in design['candidates'] if c.id == candidate_id), None)
        if candidate is None:
            raise ValueError(f"Candidate with ID {candidate_id} not found")
        
        # Get the feedback
        feedback_record = self.feedback_db.get_feedback(feedback_id)
        if not feedback_record:
            raise ValueError(f"Feedback with ID {feedback_id} not found")
        
        # Generate improved RTL
        improved_candidate = self._generate_improved_rtl(
            candidate, 
            design['requirements'],
            feedback_record['feedback']
        )
        
        # Store the new candidate
        design['candidates'].append(improved_candidate)
        
        # Mark this as derived from feedback
        improved_candidate.description = f"Improved based on feedback #{feedback_id}"
        
        return improved_candidate.id
    
    def _generate_improved_rtl(self, 
                             original_candidate: RTLCandidate, 
                             requirements: DesignRequirement,
                             feedback: Dict[str, Any]) -> RTLCandidate:
        """
        Generate improved RTL based on feedback
        
        Args:
            original_candidate: Original RTL candidate
            requirements: Design requirements
            feedback: Feedback information
            
        Returns:
            Improved RTL candidate
        """
        agent_config = self.agents[AgentRole.RTL_GENERATOR]
        
        # Prepare prompt with original code and feedback
        prompt = agent_config.prompt_template.format(
            raw_requirements=requirements.raw_text,
            parsed_requirements=json.dumps(requirements.parsed_components, indent=2),
            constraints=json.dumps(requirements.constraints, indent=2),
            identified_patterns=", ".join(requirements.identified_patterns),
            target_language=original_candidate.language.value
        )
        
        # Add the original code and feedback
        prompt += f"\n\nOriginal code:\n```{original_candidate.language.value}\n{original_candidate.code}\n```\n\n"
        prompt += f"Expert feedback:\n{json.dumps(feedback, indent=2)}\n\n"
        prompt += "Please generate an improved version of the RTL code that addresses the feedback."
        
        # Get response from LLM
        messages = [LLMMessage(role="system", content=prompt)]
        response = self.llm_client.generate(
            messages=messages,
            temperature=agent_config.temperature,
            top_p=agent_config.top_p,
            max_tokens=agent_config.max_tokens,
            model=agent_config.model_type
        )
        
        # Extract code from response
        improved_code = self._extract_code_block(response.content)
        
        # Create candidate object
        improved_candidate = RTLCandidate(
            id=str(uuid.uuid4()),
            code=improved_code,
            language=original_candidate.language,
            module_name=original_candidate.module_name,
            description=f"Improved version of {original_candidate.id}",
            generator_agent_id=str(agent_config.role.value),
            generated_at=datetime.now()
        )
        
        return improved_candidate
    
    def fine_tune_agents(self, agent_role: AgentRole, min_examples: int = 50) -> bool:
        """
        Fine-tune a specific agent using collected feedback
        
        Args:
            agent_role: Role of the agent to fine-tune
            min_examples: Minimum number of examples required for fine-tuning
            
        Returns:
            True if fine-tuning was initiated, False otherwise
        """
        # Get training examples for this agent
        examples = self.feedback_db.get_training_examples(agent_role.value)
        
        if len(examples) < min_examples:
            self.logger.info(f"Not enough examples ({len(examples)}/{min_examples}) to fine-tune {agent_role.value}")
            return False
        
        # In a real implementation, this would start a fine-tuning job
        # For this example, we'll just log that it would happen
        self.logger.info(f"Would fine-tune {agent_role.value} with {len(examples)} examples")
        
        # Update agent config with "fine-tuned" status
        self.agents[agent_role].prompt_template += "\n\nThis agent has been fine-tuned with expert feedback."
        
        return True
    
    # Helper methods
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response"""
        # Look for json block
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
            
        # Look for json without block markers
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return match.group(1)
            
        # If no JSON found, return the whole text
        return text
    
    def _extract_code_block(self, text: str) -> str:
        """Extract code block from text response"""
        # Look for code block with language specifier
        for lang in ['verilog', 'systemverilog', 'vhdl']:
            match = re.search(f'```{lang}\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return match.group(1)
        
        # Look for generic code block
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
            
        # If no code block found, return the whole text
        return text
    
    def _extract_description(self, text: str) -> Optional[str]:
        """Extract description from text response"""
        # Look for description section
        match = re.search(r'Description:(.*?)(?:```|\n\n)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _extract_module_name(self, code: str, language: HDLLanguage) -> str:
        """Extract module name from code"""
        if language == HDLLanguage.VERILOG or language == HDLLanguage.SYSTEMVERILOG:
            match = re.search(r'module\s+(\w+)', code)
            if match:
                return match.group(1)
        elif language == HDLLanguage.VHDL:
            match = re.search(r'entity\s+(\w+)', code)
            if match:
                return match.group(1)
        
        return "unknown_module"


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize system
    system = RTLAgentSystem("config.json")
    
    # Example workflow
    try:
        # Create a new design project
        requirements = """
        Design a 4-bit multiplexer with the following specifications:
        - 4 input buses (data_in_0, data_in_1, data_in_2, data_in_3), each 8 bits wide
        - 2-bit selector input (sel)
        - 1 output bus (data_out), 8 bits wide
        - Active-high synchronous reset
        - Single clock domain
        """
        
        design_id = system.create_design_project("4-bit Multiplexer", requirements)
        print(f"Created design project with ID: {design_id}")
        
        # Generate RTL candidates
        candidate_ids = system.generate_rtl_candidates(design_id, count=3)
        print(f"Generated {len(candidate_ids)} candidates")
        
        # Evaluate candidates
        scores = system.evaluate_rtl_candidates(design_id)
        print("Candidate scores:")
        for cid, score in scores.items():
            print(f"  {cid}: Overall = {score.get('overall', 0):.2f}")
        
        # Rank candidates
        ranked_ids = system.rank_candidates(design_id)
        print(f"Top candidate: {ranked_ids[0]}")
        
        # Generate testbench
        testbench_id = system.generate_testbench(design_id)
        print(f"Generated testbench with ID: {testbench_id}")
        
        # Run verification
        verification = system.run_verification(design_id, testbench_id)
        print(f"Verification status: {verification['status']}")
        
        # Simulate SME feedback
        feedback = {
            "rating": 8,
            "summary": "Good implementation but could use more descriptive comments",
            "issues": [
                {"line": 10, "description": "Signal naming could be more descriptive"},
                {"line": 15, "description": "Reset handling should check for active high"}
            ],
            "corrected_code": "// This would be the corrected code provided by SME"
        }
        
        feedback_id = system.collect_sme_feedback(design_id, ranked_ids[0], feedback)
        print(f"Collected feedback with ID: {feedback_id}")
        
        # Generate improved version
        improved_id = system.improve_rtl_with_feedback(design_id, ranked_ids[0], feedback_id)
        print(f"Generated improved version with ID: {improved_id}")
        
    except Exception as e:
        print(f"Error in example workflow: {str(e)}")