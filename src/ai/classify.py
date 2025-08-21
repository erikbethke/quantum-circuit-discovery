"""
AI-powered circuit classification and labeling
Uses LLMs to understand what quantum circuits do
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import os
from datetime import datetime

# Support multiple LLM providers
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class LLMProvider(Enum):
    """Available LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # Fallback mock provider


@dataclass
class CircuitClassification:
    """Classification result for a quantum circuit"""
    circuit_id: str
    labels: List[str]  # e.g., ["entangler", "GHZ-like", "error-correcting"]
    description: str  # Human-readable description
    confidence: float  # 0-1 confidence score
    input_spec: Optional[str] = None  # Expected input format
    output_spec: Optional[str] = None  # Expected output format
    potential_applications: List[str] = field(default_factory=list)
    similar_algorithms: List[str] = field(default_factory=list)
    novel_aspects: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2)


class CircuitClassifier:
    """
    Classify quantum circuits using LLMs
    """
    
    def __init__(self, 
                 provider: LLMProvider = LLMProvider.LOCAL,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.value.upper()}_API_KEY")
        
        # Default models
        if model:
            self.model = model
        elif provider == LLMProvider.OPENAI:
            self.model = "gpt-4-turbo-preview"
        elif provider == LLMProvider.ANTHROPIC:
            self.model = "claude-3-opus-20240229"
        else:
            self.model = "mock"
            
        # Initialize client
        self.client = self._init_client()
        
        # Few-shot examples for better classification
        self.few_shot_examples = self._load_few_shot_examples()
        
    def _init_client(self):
        """Initialize LLM client"""
        if self.provider == LLMProvider.OPENAI and HAS_OPENAI:
            return openai.OpenAI(api_key=self.api_key)
        elif self.provider == LLMProvider.ANTHROPIC and HAS_ANTHROPIC:
            return anthropic.Anthropic(api_key=self.api_key)
        else:
            return None  # Mock provider
            
    def _load_few_shot_examples(self) -> List[Dict]:
        """Load few-shot examples for classification"""
        return [
            {
                "circuit": "H(0), CNOT(0,1), CNOT(0,2)",
                "metrics": {"entanglement": 0.95, "depth": 3},
                "probabilities": {"000": 0.5, "111": 0.5},
                "classification": {
                    "labels": ["GHZ-state", "entangler", "multi-qubit"],
                    "description": "Creates a 3-qubit GHZ state with maximal entanglement",
                    "applications": ["quantum communication", "error detection"]
                }
            },
            {
                "circuit": "H(0), CNOT(0,1)",
                "metrics": {"entanglement": 1.0, "depth": 2},
                "probabilities": {"00": 0.5, "11": 0.5},
                "classification": {
                    "labels": ["Bell-pair", "EPR-pair", "entangler"],
                    "description": "Creates a Bell state (EPR pair) between two qubits",
                    "applications": ["teleportation", "superdense coding", "QKD"]
                }
            },
            {
                "circuit": "H(0), H(1), CZ(0,1), H(1)",
                "metrics": {"entanglement": 0.3, "depth": 4},
                "probabilities": {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25},
                "classification": {
                    "labels": ["uniform-superposition", "QFT-component"],
                    "description": "Creates uniform superposition with phase relationships",
                    "applications": ["quantum algorithms", "phase estimation"]
                }
            }
        ]
    
    def _format_circuit_info(self, 
                            genome_str: str,
                            metrics: Dict[str, float],
                            probabilities: Dict[str, float]) -> str:
        """Format circuit information for LLM prompt"""
        # Limit probabilities to top 10 for brevity
        sorted_probs = sorted(probabilities.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:10]
        
        info = f"""
Circuit Structure:
{genome_str}

Metrics:
- Entanglement: {metrics.get('entanglement', 0):.3f}
- Depth: {metrics.get('depth', 0)}
- Two-qubit gates: {metrics.get('two_qubit_count', 0)}
- Fidelity estimate: {metrics.get('fidelity', 0):.3f}

Output Distribution (top 10):
{json.dumps(dict(sorted_probs), indent=2)}
"""
        return info
    
    def _create_classification_prompt(self,
                                     circuit_info: str) -> str:
        """Create classification prompt with few-shot examples"""
        
        prompt = """You are an expert quantum computing scientist tasked with classifying quantum circuits.

Given a quantum circuit, its metrics, and output distribution, provide a classification in JSON format.

Examples:
"""
        
        # Add few-shot examples
        for example in self.few_shot_examples[:2]:
            prompt += f"""
Input:
Circuit: {example['circuit']}
Metrics: {json.dumps(example['metrics'])}
Probabilities: {json.dumps(example['probabilities'])}

Output:
{json.dumps(example['classification'], indent=2)}
---
"""
        
        prompt += f"""
Now classify this circuit:

{circuit_info}

Provide your classification in this JSON format:
{{
    "labels": ["label1", "label2", ...],
    "description": "Clear description of what the circuit does",
    "confidence": 0.0-1.0,
    "input_spec": "Description of expected inputs",
    "output_spec": "Description of expected outputs",
    "potential_applications": ["app1", "app2", ...],
    "similar_algorithms": ["algo1", "algo2", ...],
    "novel_aspects": "What makes this circuit interesting or unique"
}}

Focus on quantum properties and potential uses. Be specific and technical.
"""
        
        return prompt
    
    async def classify_circuit(self,
                              genome_str: str,
                              metrics: Dict[str, float],
                              probabilities: Dict[str, float],
                              circuit_id: str) -> CircuitClassification:
        """
        Classify a quantum circuit using LLM
        
        Args:
            genome_str: String representation of circuit
            metrics: Circuit metrics
            probabilities: Output probability distribution
            circuit_id: Unique circuit identifier
            
        Returns:
            CircuitClassification object
        """
        
        # Format circuit information
        circuit_info = self._format_circuit_info(genome_str, metrics, probabilities)
        
        # Create prompt
        prompt = self._create_classification_prompt(circuit_info)
        
        # Get classification from LLM
        if self.provider == LLMProvider.LOCAL:
            # Mock classification for testing
            classification_json = self._mock_classification(metrics, probabilities)
        else:
            classification_json = await self._call_llm(prompt)
            
        # Parse response
        try:
            classification_data = json.loads(classification_json)
        except json.JSONDecodeError:
            # Fallback classification
            classification_data = {
                "labels": ["unknown"],
                "description": "Unable to parse LLM response",
                "confidence": 0.0
            }
            
        # Create classification object
        return CircuitClassification(
            circuit_id=circuit_id,
            labels=classification_data.get("labels", ["unknown"]),
            description=classification_data.get("description", ""),
            confidence=classification_data.get("confidence", 0.5),
            input_spec=classification_data.get("input_spec"),
            output_spec=classification_data.get("output_spec"),
            potential_applications=classification_data.get("potential_applications", []),
            similar_algorithms=classification_data.get("similar_algorithms", []),
            novel_aspects=classification_data.get("novel_aspects")
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        if self.provider == LLMProvider.OPENAI and HAS_OPENAI:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a quantum computing expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
            
        elif self.provider == LLMProvider.ANTHROPIC and HAS_ANTHROPIC:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            return response.content[0].text
            
        else:
            return self._mock_classification({}, {})
    
    def _mock_classification(self, 
                            metrics: Dict[str, float],
                            probabilities: Dict[str, float]) -> str:
        """Mock classification for testing without API"""
        
        # Simple heuristic classification
        labels = []
        description = "Quantum circuit with "
        
        entanglement = metrics.get("entanglement", 0)
        if entanglement > 0.8:
            labels.append("high-entanglement")
            description += "high entanglement"
        elif entanglement > 0.4:
            labels.append("moderate-entanglement")
            description += "moderate entanglement"
        else:
            labels.append("low-entanglement")
            description += "low entanglement"
            
        # Check for specific patterns
        if len(probabilities) == 2:
            if abs(list(probabilities.values())[0] - 0.5) < 0.1:
                labels.append("superposition")
                description += ", creates superposition"
                
        # Check for GHZ pattern
        states = list(probabilities.keys())
        if len(states) > 0:
            all_zeros = "0" * len(states[0])
            all_ones = "1" * len(states[0])
            if (probabilities.get(all_zeros, 0) > 0.4 and 
                probabilities.get(all_ones, 0) > 0.4):
                labels.append("GHZ-like")
                description += ", GHZ-like state"
                
        classification = {
            "labels": labels,
            "description": description,
            "confidence": 0.7,
            "input_spec": "Standard computational basis",
            "output_spec": "Quantum state",
            "potential_applications": ["quantum computing", "research"],
            "similar_algorithms": [],
            "novel_aspects": "Automatically discovered circuit"
        }
        
        return json.dumps(classification)
    
    async def classify_batch(self,
                            circuits: List[Dict[str, Any]]) -> List[CircuitClassification]:
        """
        Classify multiple circuits
        
        Args:
            circuits: List of circuit dictionaries with genome_str, metrics, probabilities
            
        Returns:
            List of classifications
        """
        tasks = [
            self.classify_circuit(
                c["genome_str"],
                c["metrics"],
                c["probabilities"],
                c["circuit_id"]
            )
            for c in circuits
        ]
        
        return await asyncio.gather(*tasks)
    
    def validate_classification(self, 
                               classification: CircuitClassification,
                               metrics: Dict[str, float]) -> bool:
        """
        Validate classification against metrics
        
        Args:
            classification: Classification to validate
            metrics: Circuit metrics
            
        Returns:
            True if classification seems valid
        """
        # Check for consistency
        if "high-entanglement" in classification.labels:
            if metrics.get("entanglement", 0) < 0.7:
                return False
                
        if "GHZ-like" in classification.labels:
            if metrics.get("entanglement", 0) < 0.8:
                return False
                
        if "shallow" in classification.labels:
            if metrics.get("depth", 0) > 10:
                return False
                
        return True


class ClassificationCache:
    """Cache for circuit classifications to avoid redundant API calls"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache: Dict[str, CircuitClassification] = {}
        self.cache_size = cache_size
        
    def get(self, circuit_hash: str) -> Optional[CircuitClassification]:
        """Get cached classification"""
        return self.cache.get(circuit_hash)
    
    def put(self, circuit_hash: str, classification: CircuitClassification):
        """Store classification in cache"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            
        self.cache[circuit_hash] = classification
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()