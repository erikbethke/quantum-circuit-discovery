"""
Application Mapping and Feedback System
Maps discovered circuits to potential applications and provides feedback to evolution
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd
from datetime import datetime
import hashlib

from ..ai.classify import CircuitClassification, LLMProvider


@dataclass
class KnownAlgorithm:
    """Known quantum algorithm in knowledge base"""
    name: str
    category: str  # optimization, cryptography, simulation, etc.
    description: str
    key_properties: Dict[str, Any]
    applications: List[str]
    embedding: Optional[np.ndarray] = None
    references: List[str] = field(default_factory=list)
    

@dataclass
class ApplicationMapping:
    """Mapping from circuit to application"""
    circuit_id: str
    algorithm_matches: List[Tuple[str, float]]  # (algorithm_name, similarity)
    proposed_applications: List[str]
    confidence: float
    relevance_score: float
    hardware_fit_score: float
    novelty_score: float
    combined_score: float
    rationale: str
    validation_plan: Optional[str] = None
    feedback_weights: Optional[Dict[str, float]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_json(self) -> str:
        """Convert to JSON"""
        data = asdict(self)
        # Convert numpy arrays to lists
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = data['embedding'].tolist()
        return json.dumps(data, indent=2)


class ApplicationMapper:
    """
    Map discovered circuits to applications and provide evolution feedback
    """
    
    def __init__(self, 
                 kb_path: Optional[str] = None,
                 llm_provider: LLMProvider = LLMProvider.LOCAL):
        self.kb_path = kb_path or "data/quantum_algorithms_kb.json"
        self.knowledge_base = self._load_knowledge_base()
        self.llm_provider = llm_provider
        
        # Initialize embedding index (would be FAISS in production)
        self.embeddings_index = self._build_embeddings_index()
        
        # Hardware preferences
        self.hardware_preferences = {
            "ionq": {
                "max_qubits": 32,
                "preferred_gates": ["gpi", "gpi2", "ms", "xx"],
                "connectivity": "all-to-all"
            }
        }
        
    def _load_knowledge_base(self) -> List[KnownAlgorithm]:
        """Load knowledge base of known quantum algorithms"""
        # Default algorithms if KB doesn't exist
        default_algorithms = [
            KnownAlgorithm(
                name="Grover's Algorithm",
                category="search",
                description="Quantum search algorithm with quadratic speedup",
                key_properties={
                    "speedup": "quadratic",
                    "oracle_calls": "O(sqrt(N))",
                    "entanglement": "moderate",
                    "depth": "O(sqrt(N))"
                },
                applications=["database search", "optimization", "cryptanalysis"],
                references=["arXiv:quant-ph/9605043"]
            ),
            KnownAlgorithm(
                name="Quantum Fourier Transform",
                category="transform",
                description="Quantum analog of discrete Fourier transform",
                key_properties={
                    "gates": "O(n^2)",
                    "depth": "O(n)",
                    "entanglement": "high",
                    "pattern": "sequential rotations"
                },
                applications=["phase estimation", "Shor's algorithm", "quantum simulation"],
                references=["arXiv:quant-ph/0008040"]
            ),
            KnownAlgorithm(
                name="GHZ State",
                category="entanglement",
                description="Maximally entangled multi-qubit state",
                key_properties={
                    "entanglement": "maximal",
                    "depth": "O(n)",
                    "pattern": "star topology",
                    "measurement": "all 0s or all 1s"
                },
                applications=["quantum communication", "error correction", "metrology"],
                references=["PRL 65, 1838 (1990)"]
            ),
            KnownAlgorithm(
                name="Deutsch-Jozsa Algorithm",
                category="oracle",
                description="Determines if function is constant or balanced",
                key_properties={
                    "speedup": "exponential",
                    "oracle_calls": 1,
                    "pattern": "Hadamard sandwich",
                    "deterministic": True
                },
                applications=["algorithm demonstration", "oracle discrimination"],
                references=["Proc. R. Soc. Lond. A 439, 553 (1992)"]
            ),
            KnownAlgorithm(
                name="W State",
                category="entanglement",
                description="Robust multi-qubit entangled state",
                key_properties={
                    "entanglement": "high",
                    "robustness": "single qubit loss tolerant",
                    "pattern": "equal superposition of single excitations"
                },
                applications=["quantum communication", "quantum networks"],
                references=["PRA 62, 062314 (2000)"]
            ),
            KnownAlgorithm(
                name="Variational Quantum Eigensolver",
                category="hybrid",
                description="Hybrid quantum-classical algorithm for eigenvalue problems",
                key_properties={
                    "type": "variational",
                    "depth": "shallow",
                    "parameters": "many",
                    "optimization": "classical"
                },
                applications=["quantum chemistry", "materials science", "optimization"],
                references=["Nature Communications 5, 4213 (2014)"]
            ),
            KnownAlgorithm(
                name="Quantum Approximate Optimization",
                category="optimization",
                description="Approximate optimization for combinatorial problems",
                key_properties={
                    "type": "variational",
                    "layers": "p",
                    "parameters": "2p",
                    "approximation": "bounded"
                },
                applications=["MaxCut", "graph problems", "scheduling"],
                references=["arXiv:1411.4028"]
            )
        ]
        
        # Try to load from file
        kb_file = Path(self.kb_path)
        if kb_file.exists():
            try:
                with open(kb_file, 'r') as f:
                    data = json.load(f)
                    algorithms = [KnownAlgorithm(**algo) for algo in data]
                    return algorithms
            except Exception:
                pass
                
        return default_algorithms
    
    def _build_embeddings_index(self) -> Dict[str, np.ndarray]:
        """Build embeddings index for similarity search"""
        # Simple random embeddings for demo
        # In production, use sentence transformers or LLM embeddings
        embeddings = {}
        
        for algo in self.knowledge_base:
            # Create simple embedding from properties
            embed = np.random.randn(64)  # Would be real embedding
            
            # Encode some properties
            if algo.category == "entanglement":
                embed[0] = 1.0
            elif algo.category == "search":
                embed[1] = 1.0
            elif algo.category == "optimization":
                embed[2] = 1.0
                
            embeddings[algo.name] = embed / np.linalg.norm(embed)
            
        return embeddings
    
    def find_similar_algorithms(self,
                               behavior_embedding: np.ndarray,
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar known algorithms using embedding similarity
        
        Args:
            behavior_embedding: Circuit behavior embedding
            top_k: Number of similar algorithms to return
            
        Returns:
            List of (algorithm_name, similarity_score) tuples
        """
        similarities = []
        
        for algo_name, algo_embedding in self.embeddings_index.items():
            # Cosine similarity
            similarity = np.dot(behavior_embedding, algo_embedding)
            similarities.append((algo_name, float(similarity)))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def compute_hardware_fit(self,
                            metrics: Dict[str, float],
                            backend: str = "ionq") -> float:
        """
        Compute hardware fit score
        
        Args:
            metrics: Circuit metrics
            backend: Hardware backend
            
        Returns:
            Hardware fit score (0-1)
        """
        prefs = self.hardware_preferences.get(backend, {})
        
        score = 1.0
        
        # Penalize for exceeding qubit limit
        qubit_count = metrics.get("qubit_count", 0)
        max_qubits = prefs.get("max_qubits", 32)
        if qubit_count > max_qubits:
            score *= 0.1
        else:
            score *= (1 - qubit_count / max_qubits * 0.3)
            
        # Prefer shallow circuits
        depth = metrics.get("depth", 0)
        score *= np.exp(-depth / 50)  # Exponential decay
        
        # Prefer fewer two-qubit gates
        two_qubit = metrics.get("two_qubit_count", 0)
        score *= np.exp(-two_qubit / 20)
        
        # Bonus for native gate compliance
        if metrics.get("native_compliant", False):
            score *= 1.2
            
        return min(score, 1.0)
    
    async def map_to_applications(self,
                                 classification: CircuitClassification,
                                 metrics: Dict[str, float],
                                 behavior_embedding: np.ndarray,
                                 circuit_id: str) -> ApplicationMapping:
        """
        Map circuit to potential applications
        
        Args:
            classification: Circuit classification
            metrics: Circuit metrics
            behavior_embedding: Behavior descriptor embedding
            circuit_id: Circuit identifier
            
        Returns:
            ApplicationMapping object
        """
        
        # Find similar known algorithms
        similar_algos = self.find_similar_algorithms(behavior_embedding)
        
        # Compute scores
        hardware_fit = self.compute_hardware_fit(metrics)
        novelty = metrics.get("novelty", 0.5)
        
        # Aggregate proposed applications
        proposed_apps = set(classification.potential_applications)
        
        # Add applications from similar algorithms
        for algo_name, similarity in similar_algos[:3]:
            algo = next((a for a in self.knowledge_base if a.name == algo_name), None)
            if algo and similarity > 0.5:
                proposed_apps.update(algo.applications)
                
        # Compute relevance score
        relevance = 0.0
        if similar_algos:
            # Weighted average of top similarities
            weights = [0.5, 0.3, 0.2]
            for i, (_, sim) in enumerate(similar_algos[:3]):
                relevance += sim * weights[i] if i < len(weights) else 0
                
        # Combined score for ranking
        combined = (
            0.3 * relevance +
            0.3 * hardware_fit +
            0.2 * novelty +
            0.2 * classification.confidence
        )
        
        # Generate rationale
        rationale = self._generate_rationale(
            classification, similar_algos, hardware_fit, novelty
        )
        
        # Generate validation plan
        validation_plan = self._generate_validation_plan(
            proposed_apps, metrics
        )
        
        # Compute feedback weights for evolution
        feedback_weights = self._compute_feedback_weights(
            relevance, hardware_fit, novelty, proposed_apps
        )
        
        return ApplicationMapping(
            circuit_id=circuit_id,
            algorithm_matches=similar_algos,
            proposed_applications=list(proposed_apps),
            confidence=classification.confidence,
            relevance_score=relevance,
            hardware_fit_score=hardware_fit,
            novelty_score=novelty,
            combined_score=combined,
            rationale=rationale,
            validation_plan=validation_plan,
            feedback_weights=feedback_weights
        )
    
    def _generate_rationale(self,
                           classification: CircuitClassification,
                           similar_algos: List[Tuple[str, float]],
                           hardware_fit: float,
                           novelty: float) -> str:
        """Generate rationale for mapping"""
        
        rationale = f"Circuit classified as {', '.join(classification.labels)}. "
        
        if similar_algos and similar_algos[0][1] > 0.7:
            rationale += f"Shows strong similarity to {similar_algos[0][0]} ({similar_algos[0][1]:.2f}). "
        elif similar_algos and similar_algos[0][1] > 0.4:
            rationale += f"Partially resembles {similar_algos[0][0]}. "
        else:
            rationale += "Appears to be a novel circuit pattern. "
            
        if hardware_fit > 0.8:
            rationale += "Excellent fit for IonQ hardware. "
        elif hardware_fit > 0.5:
            rationale += "Good hardware compatibility. "
        else:
            rationale += "May require optimization for hardware. "
            
        if novelty > 0.7:
            rationale += "Highly novel behavior not seen in archive. "
            
        return rationale
    
    def _generate_validation_plan(self,
                                 applications: set,
                                 metrics: Dict[str, float]) -> str:
        """Generate validation plan for proposed applications"""
        
        plan = "Validation steps:\n"
        
        if "optimization" in str(applications):
            plan += "1. Test on small MaxCut instances\n"
            plan += "2. Compare to classical heuristics\n"
            
        if "communication" in str(applications):
            plan += "1. Verify entanglement distribution\n"
            plan += "2. Test teleportation fidelity\n"
            
        if "error" in str(applications):
            plan += "1. Simulate error injection\n"
            plan += "2. Measure correction success rate\n"
            
        plan += f"3. Run on IonQ hardware with {metrics.get('qubit_count', 3)} qubits\n"
        plan += "4. Statistical validation with 1000+ shots\n"
        
        return plan
    
    def _compute_feedback_weights(self,
                                 relevance: float,
                                 hardware_fit: float,
                                 novelty: float,
                                 applications: set) -> Dict[str, float]:
        """
        Compute feedback weights to bias evolution
        
        Returns:
            Dictionary of weight adjustments for fitness functions
        """
        weights = {}
        
        # If high relevance, encourage similar circuits
        if relevance > 0.7:
            weights["entanglement"] = 0.1  # Increase if entanglement-based app
            
        # If poor hardware fit, encourage efficiency
        if hardware_fit < 0.5:
            weights["depth_efficiency"] = 0.2
            weights["gate_efficiency"] = 0.2
            
        # If highly novel, encourage more exploration
        if novelty > 0.8:
            weights["novelty"] = 0.15
            
        # Application-specific biases
        if "optimization" in str(applications):
            weights["depth_efficiency"] = 0.1
            
        if "communication" in str(applications):
            weights["entanglement"] = 0.2
            
        return weights
    
    def update_knowledge_base(self,
                             circuit_id: str,
                             classification: CircuitClassification,
                             mapping: ApplicationMapping,
                             metrics: Dict[str, float]):
        """
        Update knowledge base with newly discovered algorithm
        
        Args:
            circuit_id: Circuit identifier
            classification: Circuit classification
            mapping: Application mapping
            metrics: Circuit metrics
        """
        
        if mapping.combined_score > 0.8 and mapping.novelty_score > 0.7:
            # This is a significant discovery
            new_algo = KnownAlgorithm(
                name=f"DFAL_{circuit_id[:8]}",
                category="discovered",
                description=classification.description,
                key_properties={
                    "entanglement": metrics.get("entanglement", 0),
                    "depth": metrics.get("depth", 0),
                    "novelty": mapping.novelty_score,
                    "discovered_by": "DFAL"
                },
                applications=mapping.proposed_applications,
                references=[f"DFAL discovery {datetime.now().isoformat()}"]
            )
            
            self.knowledge_base.append(new_algo)
            
            # Save to file
            self._save_knowledge_base()
    
    def _save_knowledge_base(self):
        """Save knowledge base to file"""
        kb_file = Path(self.kb_path)
        kb_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for algo in self.knowledge_base:
            algo_dict = asdict(algo)
            # Remove numpy arrays
            if 'embedding' in algo_dict:
                del algo_dict['embedding']
            data.append(algo_dict)
            
        with open(kb_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_application_statistics(self) -> Dict[str, Any]:
        """Get statistics about mapped applications"""
        stats = {
            "total_algorithms": len(self.knowledge_base),
            "categories": {},
            "applications": {}
        }
        
        for algo in self.knowledge_base:
            # Count by category
            if algo.category not in stats["categories"]:
                stats["categories"][algo.category] = 0
            stats["categories"][algo.category] += 1
            
            # Count by application
            for app in algo.applications:
                if app not in stats["applications"]:
                    stats["applications"][app] = 0
                stats["applications"][app] += 1
                
        return stats