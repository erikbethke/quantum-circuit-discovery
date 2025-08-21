"""
Quantum Circuit Genome Representation
Core data structures for evolutionary quantum circuit discovery
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
from enum import Enum
import numpy as np
import uuid
from datetime import datetime


class GateType(Enum):
    """IonQ Native Gate Set + Common Gates"""
    # IonQ Native Gates
    GPI = "gpi"
    GPI2 = "gpi2"
    MS = "ms"  # Mølmer-Sørensen gate
    XX = "xx"  # IonQ's XX gate
    
    # Standard Gates (for initial exploration)
    H = "h"
    X = "x"
    Y = "y"
    Z = "z"
    CNOT = "cnot"
    RZ = "rz"
    RY = "ry"
    RX = "rx"
    
    # Composite Primitives (discovered patterns)
    BELL_PAIR = "bell_pair"
    GHZ_BLOCK = "ghz_block"
    QFT_LAYER = "qft_layer"


@dataclass
class Gene:
    """Single quantum gate or operation"""
    gate_type: GateType
    targets: List[int]
    controls: Optional[List[int]] = None
    parameters: Optional[Dict[str, float]] = None
    
    def to_ionq_format(self) -> Dict:
        """Convert to IonQ API format"""
        gate_dict = {"gate": self.gate_type.value}
        
        if self.gate_type in [GateType.H, GateType.X, GateType.Y, GateType.Z]:
            gate_dict["target"] = self.targets[0]
        elif self.gate_type == GateType.CNOT:
            gate_dict["control"] = self.controls[0]
            gate_dict["target"] = self.targets[0]
        elif self.gate_type in [GateType.GPI, GateType.GPI2]:
            gate_dict["target"] = self.targets[0]
            gate_dict["phase"] = self.parameters.get("phase", 0)
        elif self.gate_type == GateType.MS:
            gate_dict["targets"] = self.targets
            gate_dict["phases"] = self.parameters.get("phases", [0, 0])
        elif self.gate_type in [GateType.RX, GateType.RY, GateType.RZ]:
            gate_dict["target"] = self.targets[0]
            gate_dict["rotation"] = self.parameters.get("angle", 0)
            
        return gate_dict


@dataclass
class QuantumGenome:
    """Complete quantum circuit genome"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    genes: List[Gene] = field(default_factory=list)
    qubit_count: int = 3
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    # Hardware targeting
    hardware_target: str = "simulator"  # "simulator", "aria-1", "forte-1"
    native_gate_compliant: bool = False
    
    # Fitness scores
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    novelty_score: float = 0.0
    
    # Behavioral descriptors for MAP-Elites
    behavior_descriptor: Optional[np.ndarray] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    discovery_notes: str = ""
    classification: Optional[str] = None
    
    def to_circuit_json(self) -> Dict:
        """Convert to IonQ circuit format"""
        return {
            "format": "ionq.circuit.v0",
            "qubits": self.qubit_count,
            "circuit": [gene.to_ionq_format() for gene in self.genes],
            "gateset": "native" if self.native_gate_compliant else None
        }
    
    def depth(self) -> int:
        """Calculate circuit depth"""
        if not self.genes:
            return 0
        
        qubit_timeline = [0] * self.qubit_count
        
        for gene in self.genes:
            affected_qubits = gene.targets + (gene.controls or [])
            max_time = max(qubit_timeline[q] for q in affected_qubits)
            for q in affected_qubits:
                qubit_timeline[q] = max_time + 1
                
        return max(qubit_timeline)
    
    def two_qubit_gate_count(self) -> int:
        """Count two-qubit gates"""
        two_qubit_gates = [GateType.CNOT, GateType.MS, GateType.XX]
        return sum(1 for g in self.genes if g.gate_type in two_qubit_gates)
    
    def clone(self) -> 'QuantumGenome':
        """Deep copy of genome"""
        import copy
        return copy.deepcopy(self)
    
    def validate(self) -> bool:
        """Check if genome is valid"""
        for gene in self.genes:
            # Check qubit bounds
            all_qubits = gene.targets + (gene.controls or [])
            if any(q >= self.qubit_count or q < 0 for q in all_qubits):
                return False
                
            # Check native gate compliance
            if self.native_gate_compliant:
                allowed = [GateType.GPI, GateType.GPI2, GateType.MS, GateType.XX]
                if gene.gate_type not in allowed:
                    return False
                    
        return True
    
    def to_qiskit_circuit(self):
        """Convert to Qiskit QuantumCircuit for local simulation"""
        try:
            from qiskit import QuantumCircuit
            
            qc = QuantumCircuit(self.qubit_count)
            
            for gene in self.genes:
                if gene.gate_type == GateType.H:
                    qc.h(gene.targets[0])
                elif gene.gate_type == GateType.X:
                    qc.x(gene.targets[0])
                elif gene.gate_type == GateType.Y:
                    qc.y(gene.targets[0])
                elif gene.gate_type == GateType.Z:
                    qc.z(gene.targets[0])
                elif gene.gate_type == GateType.CNOT:
                    qc.cnot(gene.controls[0], gene.targets[0])
                elif gene.gate_type == GateType.RX:
                    qc.rx(gene.parameters["angle"], gene.targets[0])
                elif gene.gate_type == GateType.RY:
                    qc.ry(gene.parameters["angle"], gene.targets[0])
                elif gene.gate_type == GateType.RZ:
                    qc.rz(gene.parameters["angle"], gene.targets[0])
                # Add more gate mappings as needed
                    
            return qc
        except ImportError:
            raise ImportError("Qiskit not installed. Run: pip install qiskit")


class GenomeFactory:
    """Factory for creating quantum genomes"""
    
    @staticmethod
    def create_random(qubit_count: int = 3, 
                      max_gates: int = 10,
                      native_only: bool = False) -> QuantumGenome:
        """Create a random genome"""
        genome = QuantumGenome(qubit_count=qubit_count)
        
        if native_only:
            genome.native_gate_compliant = True
            gate_types = [GateType.GPI, GateType.GPI2, GateType.MS]
        else:
            gate_types = [GateType.H, GateType.X, GateType.CNOT, 
                         GateType.RZ, GateType.RY]
        
        num_gates = np.random.randint(1, max_gates + 1)
        
        for _ in range(num_gates):
            gate_type = np.random.choice(gate_types)
            
            if gate_type == GateType.CNOT:
                targets = [np.random.randint(0, qubit_count)]
                controls = [np.random.randint(0, qubit_count)]
                while controls[0] == targets[0]:
                    controls = [np.random.randint(0, qubit_count)]
                gene = Gene(gate_type, targets, controls)
            elif gate_type == GateType.MS:
                q1, q2 = np.random.choice(qubit_count, 2, replace=False)
                gene = Gene(gate_type, [q1, q2], 
                           parameters={"phases": [0, 0]})
            elif gate_type in [GateType.GPI, GateType.GPI2]:
                target = np.random.randint(0, qubit_count)
                phase = np.random.uniform(0, 2*np.pi)
                gene = Gene(gate_type, [target], 
                           parameters={"phase": phase})
            elif gate_type in [GateType.RX, GateType.RY, GateType.RZ]:
                target = np.random.randint(0, qubit_count)
                angle = np.random.uniform(0, 2*np.pi)
                gene = Gene(gate_type, [target], 
                           parameters={"angle": angle})
            else:
                target = np.random.randint(0, qubit_count)
                gene = Gene(gate_type, [target])
                
            genome.genes.append(gene)
            
        return genome
    
    @staticmethod
    def create_bell_pair(qubit1: int = 0, qubit2: int = 1) -> QuantumGenome:
        """Create a Bell pair circuit"""
        genome = QuantumGenome(qubit_count=max(qubit1, qubit2) + 1)
        genome.genes = [
            Gene(GateType.H, [qubit1]),
            Gene(GateType.CNOT, [qubit2], [qubit1])
        ]
        genome.discovery_notes = "Bell pair - maximally entangled state"
        return genome
    
    @staticmethod
    def create_ghz(qubit_count: int = 3) -> QuantumGenome:
        """Create a GHZ state circuit"""
        genome = QuantumGenome(qubit_count=qubit_count)
        genome.genes = [Gene(GateType.H, [0])]
        
        for i in range(1, qubit_count):
            genome.genes.append(Gene(GateType.CNOT, [i], [0]))
            
        genome.discovery_notes = f"{qubit_count}-qubit GHZ state"
        return genome
    
    @staticmethod
    def create_qft(qubit_count: int = 3) -> QuantumGenome:
        """Create a Quantum Fourier Transform circuit"""
        genome = QuantumGenome(qubit_count=qubit_count)
        
        for j in range(qubit_count):
            genome.genes.append(Gene(GateType.H, [j]))
            for k in range(j + 1, qubit_count):
                angle = np.pi / (2 ** (k - j))
                genome.genes.append(
                    Gene(GateType.RZ, [k], [j], {"angle": angle})
                )
                
        # Swap qubits to get correct output order
        for i in range(qubit_count // 2):
            # Simplified swap using 3 CNOTs
            genome.genes.extend([
                Gene(GateType.CNOT, [i], [qubit_count - 1 - i]),
                Gene(GateType.CNOT, [qubit_count - 1 - i], [i]),
                Gene(GateType.CNOT, [i], [qubit_count - 1 - i])
            ])
            
        genome.discovery_notes = f"{qubit_count}-qubit QFT"
        return genome