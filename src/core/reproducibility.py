"""
Reproducibility Bundle System
Ensures every discovered circuit can be exactly reproduced
"""

import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import numpy as np
import random


@dataclass
class ReproducibilityBundle:
    """Complete bundle for reproducing a circuit discovery"""
    
    # Circuit identity
    circuit_id: str
    circuit_hash: str
    generation: int
    
    # Exact configuration
    random_seed: int
    numpy_seed: int
    evolution_config: Dict[str, Any]
    
    # Circuit definition
    genome_json: Dict[str, Any]
    gate_sequence: List[Dict[str, Any]]
    qubit_count: int
    
    # Execution details
    backend: str
    shots: int
    job_id: Optional[str] = None
    api_version: Optional[str] = None
    noise_model: Optional[Dict[str, Any]] = None
    
    # Results
    probabilities: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    behavior_descriptor: Optional[List[float]] = None
    
    # Classification
    classification: Optional[Dict[str, Any]] = None
    application_mapping: Optional[Dict[str, Any]] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dfal_version: str = "0.1.0"
    dependencies: Dict[str, str] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        data = asdict(self)
        # Convert numpy arrays to lists
        if self.behavior_descriptor is not None:
            data['behavior_descriptor'] = list(self.behavior_descriptor)
        return json.dumps(data, indent=2)
    
    def save(self, directory: str = "reproductions"):
        """Save bundle to disk"""
        path = Path(directory) / f"{self.circuit_id}"
        path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON version
        json_file = path / "bundle.json"
        with open(json_file, 'w') as f:
            f.write(self.to_json())
            
        # Save pickle version (preserves numpy arrays)
        pickle_file = path / "bundle.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)
            
        # Save human-readable summary
        summary_file = path / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(self._create_summary())
            
        return path
    
    def _create_summary(self) -> str:
        """Create human-readable summary"""
        summary = f"""
DFAL Reproducibility Bundle
===========================
Circuit ID: {self.circuit_id}
Hash: {self.circuit_hash}
Generated: {self.timestamp}

Configuration:
- Generation: {self.generation}
- Backend: {self.backend}
- Shots: {self.shots}
- Qubits: {self.qubit_count}
- Gates: {len(self.gate_sequence)}
- Random Seed: {self.random_seed}

Metrics:
"""
        for key, value in self.metrics.items():
            summary += f"- {key}: {value:.4f}\n"
            
        if self.classification:
            summary += f"\nClassification:\n"
            summary += f"- Labels: {self.classification.get('labels', [])}\n"
            summary += f"- Description: {self.classification.get('description', '')}\n"
            
        if self.job_id:
            summary += f"\nIonQ Job ID: {self.job_id}\n"
            
        return summary
    
    @classmethod
    def load(cls, filepath: str) -> 'ReproducibilityBundle':
        """Load bundle from disk"""
        path = Path(filepath)
        
        if path.is_dir():
            # Load from directory
            pickle_file = path / "bundle.pkl"
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    return pickle.load(f)
            else:
                json_file = path / "bundle.json"
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    return cls(**data)
        else:
            # Load single file
            if filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    return cls(**data)


class ReproducibilityManager:
    """Manages reproducibility for entire evolution run"""
    
    def __init__(self, base_dir: str = "reproductions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Master seed management
        self.master_seed = None
        self.generation_seeds = {}
        
        # Version tracking
        self.versions = self._capture_versions()
        
        # Bundle cache
        self.bundles = {}
        
    def _capture_versions(self) -> Dict[str, str]:
        """Capture dependency versions"""
        versions = {
            "python": "3.10.0",  # Would get from sys.version
            "dfal": "0.1.0"
        }
        
        try:
            import numpy
            versions["numpy"] = numpy.__version__
        except:
            pass
            
        try:
            import qiskit
            versions["qiskit"] = qiskit.__version__
        except:
            pass
            
        return versions
    
    def set_master_seed(self, seed: Optional[int] = None) -> int:
        """Set master random seed"""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
            
        self.master_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        return seed
    
    def get_generation_seed(self, generation: int) -> int:
        """Get deterministic seed for generation"""
        if generation not in self.generation_seeds:
            # Derive from master seed
            gen_seed = (self.master_seed + generation * 1337) % (2**31)
            self.generation_seeds[generation] = gen_seed
            
        return self.generation_seeds[generation]
    
    def create_bundle(self,
                     genome,
                     generation: int,
                     backend: str = "simulator",
                     shots: int = 1000) -> ReproducibilityBundle:
        """Create reproducibility bundle for a genome"""
        
        # Compute circuit hash
        circuit_str = json.dumps(genome.to_circuit_json(), sort_keys=True)
        circuit_hash = hashlib.sha256(circuit_str.encode()).hexdigest()[:16]
        
        # Get seeds
        gen_seed = self.get_generation_seed(generation)
        
        # Create bundle
        bundle = ReproducibilityBundle(
            circuit_id=genome.id,
            circuit_hash=circuit_hash,
            generation=generation,
            random_seed=gen_seed,
            numpy_seed=gen_seed,
            evolution_config={
                "master_seed": self.master_seed,
                "generation": generation
            },
            genome_json=genome.to_circuit_json(),
            gate_sequence=[
                {
                    "gate": g.gate_type.value,
                    "targets": g.targets,
                    "controls": g.controls,
                    "parameters": g.parameters
                }
                for g in genome.genes
            ],
            qubit_count=genome.qubit_count,
            backend=backend,
            shots=shots,
            dependencies=self.versions
        )
        
        # Cache bundle
        self.bundles[genome.id] = bundle
        
        return bundle
    
    def update_bundle_results(self,
                             circuit_id: str,
                             probabilities: Dict[str, float],
                             metrics: Dict[str, float],
                             job_id: Optional[str] = None):
        """Update bundle with execution results"""
        if circuit_id in self.bundles:
            bundle = self.bundles[circuit_id]
            bundle.probabilities = probabilities
            bundle.metrics = metrics
            bundle.job_id = job_id
            
    def update_bundle_classification(self,
                                    circuit_id: str,
                                    classification: Dict[str, Any],
                                    mapping: Optional[Dict[str, Any]] = None):
        """Update bundle with classification results"""
        if circuit_id in self.bundles:
            bundle = self.bundles[circuit_id]
            bundle.classification = classification
            bundle.application_mapping = mapping
            
    def save_generation(self, generation: int):
        """Save all bundles from a generation"""
        gen_dir = self.base_dir / f"generation_{generation:04d}"
        gen_dir.mkdir(exist_ok=True)
        
        saved = 0
        for circuit_id, bundle in self.bundles.items():
            if bundle.generation == generation:
                bundle.save(str(gen_dir))
                saved += 1
                
        # Save generation metadata
        metadata = {
            "generation": generation,
            "seed": self.get_generation_seed(generation),
            "circuits_saved": saved,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(gen_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return gen_dir
    
    def replay_circuit(self, bundle: ReproducibilityBundle):
        """Replay a circuit from bundle"""
        # Set seeds
        random.seed(bundle.random_seed)
        np.random.seed(bundle.numpy_seed)
        
        # Reconstruct genome
        from ..core.genome import QuantumGenome, Gene, GateType
        
        genome = QuantumGenome(
            id=bundle.circuit_id,
            qubit_count=bundle.qubit_count,
            generation=bundle.generation
        )
        
        for gate_data in bundle.gate_sequence:
            gene = Gene(
                gate_type=GateType(gate_data["gate"]),
                targets=gate_data["targets"],
                controls=gate_data["controls"],
                parameters=gate_data["parameters"]
            )
            genome.genes.append(gene)
            
        return genome
    
    def verify_reproducibility(self,
                              bundle1: ReproducibilityBundle,
                              bundle2: ReproducibilityBundle) -> bool:
        """Verify two bundles represent the same circuit"""
        return bundle1.circuit_hash == bundle2.circuit_hash
    
    def export_discoveries(self, 
                          min_fitness: float = 0.7,
                          output_file: str = "discoveries.json"):
        """Export significant discoveries"""
        discoveries = []
        
        for circuit_id, bundle in self.bundles.items():
            if bundle.metrics.get("combined_fitness", 0) >= min_fitness:
                discoveries.append({
                    "circuit_id": circuit_id,
                    "hash": bundle.circuit_hash,
                    "generation": bundle.generation,
                    "fitness": bundle.metrics.get("combined_fitness", 0),
                    "metrics": bundle.metrics,
                    "classification": bundle.classification,
                    "replay_bundle": f"generation_{bundle.generation:04d}/{circuit_id}"
                })
                
        # Sort by fitness
        discoveries.sort(key=lambda x: x["fitness"], reverse=True)
        
        with open(output_file, 'w') as f:
            json.dump(discoveries, f, indent=2)
            
        return discoveries