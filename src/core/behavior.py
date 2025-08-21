"""
Behavior Descriptors for Quantum Circuits
Defines how we characterize what a circuit does (not how well)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
from scipy.stats import entropy
from scipy.linalg import sqrtm
import warnings

warnings.filterwarnings('ignore', category=np.ComplexWarning)


@dataclass
class BehaviorMetrics:
    """Comprehensive behavior characterization"""
    probability_vector: np.ndarray  # First N probabilities
    shannon_entropy: float
    entanglement_spectrum: np.ndarray
    kl_divergence_uniform: float
    concentration: float  # Max probability
    support_size: int  # Number of non-zero outputs
    depth: int
    two_qubit_count: int
    qubit_count: int
    fidelity_estimate: float
    

class BehaviorDescriptor:
    """
    Compute behavior descriptors that characterize quantum circuit outputs
    Used for novelty search and MAP-Elites binning
    """
    
    def __init__(self, descriptor_dim: int = 64):
        self.descriptor_dim = descriptor_dim
        self.prob_vector_size = 32  # First 32 basis states
        
    def compute_descriptor(self, 
                          probabilities: Dict[str, float],
                          metrics: Dict[str, float],
                          qubit_count: int) -> np.ndarray:
        """
        Compute fixed-size behavior descriptor from circuit output
        
        Args:
            probabilities: Output probability distribution
            metrics: Circuit metrics (depth, gates, etc.)
            qubit_count: Number of qubits
            
        Returns:
            Fixed-size numpy array descriptor
        """
        # Convert to sorted probability vector
        n_states = min(2**qubit_count, self.prob_vector_size)
        prob_vector = np.zeros(n_states)
        
        for i in range(n_states):
            bitstring = format(i, f'0{qubit_count}b')
            prob_vector[i] = probabilities.get(bitstring, 0.0)
            
        # Compute entropy
        H = self._shannon_entropy(prob_vector)
        
        # Compute KL divergence from uniform
        kl_uniform = self._kl_divergence_uniform(probabilities, qubit_count)
        
        # Concentration (max probability)
        concentration = max(probabilities.values()) if probabilities else 0
        
        # Support size (non-zero outputs)
        support = sum(1 for p in probabilities.values() if p > 0.01)
        
        # Build descriptor
        descriptor = np.concatenate([
            prob_vector[:self.prob_vector_size],  # Probability fingerprint
            [
                H,  # Shannon entropy
                kl_uniform,  # KL divergence from uniform
                concentration,  # Max probability
                support / (2**qubit_count),  # Normalized support size
                metrics.get('entanglement', 0),
                metrics.get('depth', 0) / 30.0,  # Normalized depth
                metrics.get('two_qubit_count', 0) / 20.0,  # Normalized 2Q gates
                metrics.get('fidelity', 0),
                qubit_count / 10.0  # Normalized qubit count
            ]
        ])
        
        # Pad or truncate to fixed size
        if len(descriptor) < self.descriptor_dim:
            descriptor = np.pad(descriptor, (0, self.descriptor_dim - len(descriptor)))
        else:
            descriptor = descriptor[:self.descriptor_dim]
            
        return descriptor
    
    def compute_entanglement_spectrum(self,
                                     state_vector: np.ndarray,
                                     qubit_count: int,
                                     partition: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute entanglement spectrum for bipartition
        
        Args:
            state_vector: Quantum state vector
            qubit_count: Total number of qubits
            partition: Qubits in subsystem A (default: first half)
            
        Returns:
            Entanglement spectrum (eigenvalues of reduced density matrix)
        """
        if partition is None:
            partition = list(range(qubit_count // 2))
            
        n_A = len(partition)
        n_B = qubit_count - n_A
        
        if n_A == 0 or n_B == 0:
            return np.array([1.0])  # No entanglement possible
            
        # Reshape state vector
        dim_A = 2**n_A
        dim_B = 2**n_B
        
        try:
            psi = state_vector.reshape(dim_A, dim_B)
            
            # Compute reduced density matrix
            rho_A = np.dot(psi, psi.conj().T)
            
            # Get eigenvalues
            eigenvalues = np.linalg.eigvalsh(rho_A)
            
            # Filter numerical zeros and sort
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            return eigenvalues
            
        except Exception:
            return np.array([1.0])
    
    def compute_bipartite_entanglement(self,
                                      state_vector: np.ndarray,
                                      qubit_count: int) -> float:
        """
        Compute average bipartite entanglement across all single-qubit partitions
        """
        if len(state_vector) != 2**qubit_count:
            return 0.0
            
        total_entanglement = 0.0
        
        for i in range(qubit_count):
            # Partition with qubit i on one side
            spectrum = self.compute_entanglement_spectrum(
                state_vector, qubit_count, partition=[i]
            )
            
            # Von Neumann entropy from spectrum
            entropy = -np.sum(spectrum * np.log2(spectrum + 1e-10))
            total_entanglement += entropy
            
        return total_entanglement / qubit_count
    
    def _shannon_entropy(self, probs: np.ndarray) -> float:
        """Compute Shannon entropy"""
        probs_nonzero = probs[probs > 1e-10]
        if len(probs_nonzero) == 0:
            return 0.0
        return -np.sum(probs_nonzero * np.log2(probs_nonzero))
    
    def _kl_divergence_uniform(self, 
                               probabilities: Dict[str, float],
                               qubit_count: int) -> float:
        """Compute KL divergence from uniform distribution"""
        n_states = 2**qubit_count
        uniform_prob = 1.0 / n_states
        
        kl = 0.0
        for state in probabilities:
            p = probabilities[state]
            if p > 1e-10:
                kl += p * np.log2(p / uniform_prob)
                
        return kl
    
    def compute_fingerprint(self, descriptor: np.ndarray) -> str:
        """
        Compute hash fingerprint of behavior descriptor
        Used for exact duplicate detection
        """
        # Round to avoid floating point issues
        rounded = np.round(descriptor, decimals=6)
        
        # Create hash
        hash_obj = hashlib.sha256(rounded.tobytes())
        return hash_obj.hexdigest()[:16]
    
    def distance(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """
        Compute distance between two behavior descriptors
        Used for novelty computation
        """
        return np.linalg.norm(desc1 - desc2)
    
    def embed_descriptor(self, descriptor: np.ndarray, 
                        method: str = 'pca',
                        target_dim: int = 16) -> np.ndarray:
        """
        Embed high-dimensional descriptor to lower dimension
        For efficient similarity search
        
        Args:
            descriptor: High-dimensional behavior descriptor
            method: Embedding method ('pca', 'umap', 'none')
            target_dim: Target dimensionality
            
        Returns:
            Lower-dimensional embedding
        """
        if method == 'none' or len(descriptor) <= target_dim:
            return descriptor
            
        if method == 'pca':
            # Simple PCA via SVD
            # In production, use sklearn PCA fitted on population
            U, s, Vt = np.linalg.svd(descriptor.reshape(1, -1), full_matrices=False)
            return s[:target_dim]
            
        # For other methods, would need sklearn/umap libraries
        return descriptor[:target_dim]


class NoveltyComputer:
    """
    Compute novelty scores using behavior descriptors
    """
    
    def __init__(self, k_nearest: int = 15):
        self.k_nearest = k_nearest
        self.archive: List[np.ndarray] = []
        self.index = None  # Would be FAISS index in production
        
    def add_to_archive(self, descriptor: np.ndarray):
        """Add descriptor to novelty archive"""
        self.archive.append(descriptor.copy())
        
        # In production, update FAISS index here
        if self.index is not None:
            self.index.add(descriptor.reshape(1, -1))
            
    def compute_novelty(self, descriptor: np.ndarray) -> float:
        """
        Compute novelty as average distance to k-nearest neighbors
        """
        if len(self.archive) == 0:
            return float('inf')  # First individual is maximally novel
            
        # Compute distances to all archived behaviors
        distances = []
        for archived in self.archive:
            dist = np.linalg.norm(descriptor - archived)
            distances.append(dist)
            
        # Sort and take k-nearest
        distances.sort()
        k = min(self.k_nearest, len(distances))
        nearest_distances = distances[:k]
        
        return np.mean(nearest_distances)
    
    def compute_novelty_batch(self, descriptors: List[np.ndarray]) -> List[float]:
        """Compute novelty for multiple descriptors efficiently"""
        return [self.compute_novelty(d) for d in descriptors]
    
    def get_diverse_subset(self, n: int) -> List[np.ndarray]:
        """
        Get n most diverse descriptors from archive
        Uses greedy max-min selection
        """
        if len(self.archive) <= n:
            return self.archive.copy()
            
        selected = []
        remaining = list(range(len(self.archive)))
        
        # Start with random
        idx = np.random.choice(remaining)
        selected.append(self.archive[idx])
        remaining.remove(idx)
        
        # Greedily select most distant
        while len(selected) < n and remaining:
            max_min_dist = -1
            max_idx = None
            
            for idx in remaining:
                candidate = self.archive[idx]
                
                # Min distance to selected
                min_dist = min(
                    np.linalg.norm(candidate - s) 
                    for s in selected
                )
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    max_idx = idx
                    
            if max_idx is not None:
                selected.append(self.archive[max_idx])
                remaining.remove(max_idx)
                
        return selected


def create_faiss_index(dimension: int, metric: str = 'L2'):
    """
    Create FAISS index for efficient similarity search
    Requires: pip install faiss-cpu
    """
    try:
        import faiss
        
        if metric == 'L2':
            index = faiss.IndexFlatL2(dimension)
        elif metric == 'IP':  # Inner product
            index = faiss.IndexFlatIP(dimension)
        else:
            index = faiss.IndexFlatL2(dimension)
            
        # Could also use approximate index for large scale
        # index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
        
        return index
        
    except ImportError:
        print("FAISS not installed. Using numpy for similarity search.")
        return None


class BehaviorAnalyzer:
    """
    Analyze and validate behavior descriptors
    """
    
    @staticmethod
    def validate_ghz(probabilities: Dict[str, float], 
                    qubit_count: int) -> bool:
        """Check if output matches GHZ pattern"""
        all_zeros = '0' * qubit_count
        all_ones = '1' * qubit_count
        
        p_zeros = probabilities.get(all_zeros, 0)
        p_ones = probabilities.get(all_ones, 0)
        
        # GHZ should have ~0.5 probability for |000...0> and |111...1>
        return (abs(p_zeros - 0.5) < 0.1 and 
                abs(p_ones - 0.5) < 0.1 and
                (p_zeros + p_ones) > 0.9)
    
    @staticmethod
    def validate_bell(probabilities: Dict[str, float]) -> bool:
        """Check if output matches Bell state pattern"""
        # Bell state: (|00> + |11>)/sqrt(2)
        p_00 = probabilities.get('00', 0)
        p_11 = probabilities.get('11', 0)
        
        return (abs(p_00 - 0.5) < 0.1 and 
                abs(p_11 - 0.5) < 0.1 and
                (p_00 + p_11) > 0.9)
    
    @staticmethod
    def validate_uniform(probabilities: Dict[str, float],
                        qubit_count: int,
                        tolerance: float = 0.1) -> bool:
        """Check if output is approximately uniform"""
        expected = 1.0 / (2**qubit_count)
        
        for prob in probabilities.values():
            if abs(prob - expected) > tolerance:
                return False
        return True