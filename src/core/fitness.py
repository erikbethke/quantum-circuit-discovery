"""
Multi-objective Fitness Evaluation for Quantum Circuits
Computes entanglement, novelty, hardware fidelity, and other quantum metrics
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import scipy.linalg as la
from collections import Counter
import hashlib
import json

from genome import QuantumGenome


@dataclass
class FitnessMetrics:
    """Container for all fitness metrics"""
    entanglement: float = 0.0
    novelty: float = 0.0
    depth_efficiency: float = 0.0
    gate_efficiency: float = 0.0
    hardware_fidelity: float = 0.0
    output_complexity: float = 0.0
    quantum_volume: float = 0.0
    behavior_vector: Optional[np.ndarray] = None
    
    def combined_score(self, weights: Dict[str, float]) -> float:
        """Compute weighted combination of metrics"""
        score = 0.0
        score += weights.get("entanglement", 0.3) * self.entanglement
        score += weights.get("novelty", 0.3) * self.novelty
        score += weights.get("depth_efficiency", -0.1) * self.depth_efficiency
        score += weights.get("gate_efficiency", -0.1) * self.gate_efficiency
        score += weights.get("hardware_fidelity", 0.2) * self.hardware_fidelity
        score += weights.get("output_complexity", 0.1) * self.output_complexity
        return score


class QuantumMetrics:
    """Quantum-specific metrics computation"""
    
    @staticmethod
    def compute_entanglement_entropy(state_vector: np.ndarray, 
                                    qubit_count: int) -> float:
        """
        Compute entanglement entropy using von Neumann entropy
        of the reduced density matrix
        """
        if len(state_vector) != 2**qubit_count:
            return 0.0
            
        # Reshape state vector to matrix form
        dim_a = 2**(qubit_count // 2)
        dim_b = 2**(qubit_count - qubit_count // 2)
        
        try:
            psi = state_vector.reshape(dim_a, dim_b)
            
            # Compute reduced density matrix for subsystem A
            rho_a = np.dot(psi, psi.conj().T)
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(rho_a)
            
            # Filter out numerical zeros
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            # Compute von Neumann entropy
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
            
            # Normalize to [0, 1]
            max_entropy = min(qubit_count // 2, qubit_count - qubit_count // 2)
            return min(entropy / max_entropy, 1.0)
            
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_meyer_wallach_entanglement(state_vector: np.ndarray,
                                          qubit_count: int) -> float:
        """
        Compute Meyer-Wallach entanglement measure
        Average entanglement across all bipartitions
        """
        if len(state_vector) != 2**qubit_count:
            return 0.0
            
        total_entanglement = 0.0
        
        for i in range(qubit_count):
            # Create bipartition with qubit i on one side
            # This is a simplified version
            dim_a = 2
            dim_b = 2**(qubit_count - 1)
            
            # Reorder state vector for this partition
            # (simplified - actual implementation would properly trace out)
            try:
                psi_reshaped = state_vector.reshape(dim_a, dim_b)
                rho_reduced = np.dot(psi_reshaped, psi_reshaped.conj().T)
                
                # Compute purity
                purity = np.real(np.trace(np.dot(rho_reduced, rho_reduced)))
                
                # Linear entropy (related to entanglement)
                linear_entropy = 1 - purity
                total_entanglement += linear_entropy
                
            except Exception:
                continue
                
        # Average over all single-qubit partitions
        return total_entanglement / qubit_count if qubit_count > 0 else 0.0
    
    @staticmethod
    def compute_output_complexity(probabilities: Dict[str, float]) -> float:
        """
        Compute Shannon entropy of output distribution
        Higher entropy = more complex/random output
        """
        if not probabilities:
            return 0.0
            
        # Convert to probability array
        probs = np.array(list(probabilities.values()))
        
        # Filter out zeros
        probs = probs[probs > 1e-10]
        
        if len(probs) == 0:
            return 0.0
            
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(probabilities))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    @staticmethod
    def compute_quantum_volume(genome: QuantumGenome) -> float:
        """
        Estimate quantum volume - combines circuit width and depth
        in a way that captures computational power
        """
        width = genome.qubit_count
        depth = genome.depth()
        
        # Simplified quantum volume metric
        # Real QV requires successful execution of random circuits
        effective_size = min(width, depth)
        
        # Normalize to [0, 1] assuming max QV of 2^10
        return min(2**effective_size / 1024, 1.0)


class BehaviorDescriptor:
    """Compute behavior descriptors for novelty search and MAP-Elites"""
    
    @staticmethod
    def compute_descriptor(genome: QuantumGenome, 
                          simulation_result: Optional[Dict] = None) -> np.ndarray:
        """
        Compute a behavior descriptor vector that characterizes
        what the circuit does, not how well it does it
        """
        descriptor = []
        
        # Structural features
        descriptor.append(genome.depth() / 30.0)  # Normalized depth
        descriptor.append(genome.two_qubit_gate_count() / 20.0)  # Normalized 2Q gates
        descriptor.append(genome.qubit_count / 10.0)  # Normalized qubit count
        
        # Gate type distribution
        gate_counts = Counter(g.gate_type.value for g in genome.genes)
        gate_types = ['h', 'cnot', 'x', 'y', 'z', 'rz', 'ry', 'rx', 'ms', 'gpi', 'gpi2']
        for gate in gate_types:
            descriptor.append(gate_counts.get(gate, 0) / max(len(genome.genes), 1))
            
        # Output distribution features (if simulation available)
        if simulation_result and 'probabilities' in simulation_result:
            probs = simulation_result['probabilities']
            
            # Number of non-zero outputs
            non_zero = sum(1 for p in probs.values() if p > 0.01)
            descriptor.append(non_zero / len(probs))
            
            # Max probability (concentration)
            descriptor.append(max(probs.values()) if probs else 0)
            
            # Entropy
            entropy = QuantumMetrics.compute_output_complexity(probs)
            descriptor.append(entropy)
            
            # First few output probabilities (fingerprint)
            sorted_states = sorted(probs.keys())[:8]
            for state in sorted_states:
                descriptor.append(probs.get(state, 0))
                
        # Pad to fixed size
        while len(descriptor) < 30:
            descriptor.append(0)
            
        return np.array(descriptor[:30])
    
    @staticmethod
    def compute_fingerprint(genome: QuantumGenome) -> str:
        """
        Compute a unique fingerprint for circuit structure
        Used for detecting exact duplicates
        """
        # Create canonical representation
        gates_str = []
        for gene in genome.genes:
            gate_repr = f"{gene.gate_type.value}"
            gate_repr += f"_t{gene.targets}"
            if gene.controls:
                gate_repr += f"_c{gene.controls}"
            if gene.parameters:
                # Round parameters to avoid floating point issues
                params_rounded = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in gene.parameters.items()
                }
                gate_repr += f"_p{params_rounded}"
            gates_str.append(gate_repr)
            
        circuit_str = f"q{genome.qubit_count}_" + "_".join(gates_str)
        
        # Create hash
        return hashlib.sha256(circuit_str.encode()).hexdigest()[:16]


class HardwareFidelityEstimator:
    """Estimate fidelity on IonQ hardware"""
    
    # IonQ gate error rates (approximate)
    GATE_ERRORS = {
        "single_qubit": 0.001,  # 0.1% error
        "two_qubit": 0.005,     # 0.5% error
        "measurement": 0.01     # 1% error
    }
    
    # IonQ gate times (microseconds)
    GATE_TIMES = {
        "gpi": 20,
        "gpi2": 20,
        "ms": 200,
        "xx": 200,
        "h": 20,
        "x": 20,
        "y": 20,
        "z": 20,
        "cnot": 200,
        "rz": 20,
        "ry": 20,
        "rx": 20
    }
    
    @classmethod
    def estimate_fidelity(cls, genome: QuantumGenome) -> float:
        """
        Estimate circuit fidelity on IonQ hardware
        Based on gate count, depth, and error rates
        """
        if not genome.genes:
            return 1.0
            
        total_error = 0.0
        total_time = 0.0
        
        for gene in genome.genes:
            # Accumulate gate errors
            if gene.gate_type.value in ['cnot', 'ms', 'xx']:
                total_error += cls.GATE_ERRORS["two_qubit"]
            else:
                total_error += cls.GATE_ERRORS["single_qubit"]
                
            # Accumulate gate times
            total_time += cls.GATE_TIMES.get(gene.gate_type.value, 50)
            
        # Add measurement error
        total_error += genome.qubit_count * cls.GATE_ERRORS["measurement"]
        
        # Decoherence error (simplified model)
        # Assuming T1 = 10ms, T2 = 1ms
        decoherence_error = total_time / 1000000  # Convert to seconds
        decoherence_error *= genome.qubit_count * 0.1  # 10% per second per qubit
        
        total_error += decoherence_error
        
        # Convert to fidelity
        fidelity = max(0, 1 - total_error)
        
        # Bonus for native gate compliance
        if genome.native_gate_compliant:
            fidelity *= 1.1  # 10% bonus
            
        return min(fidelity, 1.0)
    
    @classmethod
    def estimate_success_rate(cls, genome: QuantumGenome, shots: int = 1000) -> float:
        """
        Estimate success rate for multiple shots
        """
        single_shot_fidelity = cls.estimate_fidelity(genome)
        
        # For multiple shots, consider both systematic and statistical errors
        # This is a simplified model
        return single_shot_fidelity * (1 - 0.01 * np.log10(shots))


class FitnessEvaluator:
    """Main fitness evaluation class"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "entanglement": 0.3,
            "novelty": 0.3,
            "depth_efficiency": -0.1,
            "gate_efficiency": -0.1,
            "hardware_fidelity": 0.2,
            "output_complexity": 0.1
        }
        
        # Cache for expensive computations
        self.fingerprint_cache: Dict[str, FitnessMetrics] = {}
        
    def evaluate(self, genome: QuantumGenome, 
                simulation_result: Optional[Dict] = None) -> Tuple[float, np.ndarray]:
        """
        Evaluate a quantum genome and return fitness score and behavior descriptor
        
        Args:
            genome: The quantum circuit genome to evaluate
            simulation_result: Optional simulation results containing state vector
                             and probabilities
                             
        Returns:
            Tuple of (fitness_score, behavior_descriptor)
        """
        # Check cache
        fingerprint = BehaviorDescriptor.compute_fingerprint(genome)
        if fingerprint in self.fingerprint_cache:
            cached = self.fingerprint_cache[fingerprint]
            return cached.combined_score(self.weights), cached.behavior_vector
            
        metrics = FitnessMetrics()
        
        # Compute structural efficiency metrics
        depth = genome.depth()
        gate_count = len(genome.genes)
        two_qubit_count = genome.two_qubit_gate_count()
        
        # Normalize depth and gate efficiency (lower is better)
        metrics.depth_efficiency = 1.0 - min(depth / 30.0, 1.0)
        metrics.gate_efficiency = 1.0 - min(gate_count / 50.0, 1.0)
        
        # Hardware fidelity estimation
        metrics.hardware_fidelity = HardwareFidelityEstimator.estimate_fidelity(genome)
        
        # Quantum volume
        metrics.quantum_volume = QuantumMetrics.compute_quantum_volume(genome)
        
        # If we have simulation results, compute quantum metrics
        if simulation_result:
            if 'state_vector' in simulation_result:
                state_vector = np.array(simulation_result['state_vector'])
                
                # Entanglement metrics
                metrics.entanglement = QuantumMetrics.compute_meyer_wallach_entanglement(
                    state_vector, genome.qubit_count
                )
                
            if 'probabilities' in simulation_result:
                # Output complexity
                metrics.output_complexity = QuantumMetrics.compute_output_complexity(
                    simulation_result['probabilities']
                )
        else:
            # Estimate based on structure alone
            # Circuits with more 2-qubit gates likely create more entanglement
            metrics.entanglement = min(two_qubit_count / (genome.qubit_count * 2), 1.0)
            metrics.output_complexity = 0.5  # Neutral assumption
            
        # Compute behavior descriptor
        metrics.behavior_vector = BehaviorDescriptor.compute_descriptor(
            genome, simulation_result
        )
        
        # Note: Novelty will be computed relative to archive in evolution engine
        metrics.novelty = 0.0  # Placeholder
        
        # Cache the result
        self.fingerprint_cache[fingerprint] = metrics
        
        # Clean cache if too large
        if len(self.fingerprint_cache) > 10000:
            # Keep only recent half
            keys = list(self.fingerprint_cache.keys())
            for key in keys[:5000]:
                del self.fingerprint_cache[key]
                
        return metrics.combined_score(self.weights), metrics.behavior_vector
    
    def evaluate_batch(self, genomes: List[QuantumGenome],
                      simulation_results: Optional[List[Dict]] = None) -> List[Tuple[float, np.ndarray]]:
        """
        Evaluate a batch of genomes efficiently
        """
        results = []
        
        if simulation_results is None:
            simulation_results = [None] * len(genomes)
            
        for genome, sim_result in zip(genomes, simulation_results):
            results.append(self.evaluate(genome, sim_result))
            
        return results
    
    def get_detailed_metrics(self, genome: QuantumGenome,
                            simulation_result: Optional[Dict] = None) -> FitnessMetrics:
        """
        Get detailed metrics for analysis and visualization
        """
        fingerprint = BehaviorDescriptor.compute_fingerprint(genome)
        
        if fingerprint not in self.fingerprint_cache:
            self.evaluate(genome, simulation_result)
            
        return self.fingerprint_cache.get(fingerprint, FitnessMetrics())