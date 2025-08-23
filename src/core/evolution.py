"""
Evolutionary Engine for Quantum Circuit Discovery
Implements genetic operators, selection strategies, and population management
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
import random
from copy import deepcopy
from collections import defaultdict

from .genome import (
    QuantumGenome, Gene, GateType, GenomeFactory
)


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm"""
    population_size: int = 100
    elite_size: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 5
    max_circuit_depth: int = 20
    max_gates: int = 30
    qubit_budget: int = 5
    native_gates_only: bool = False
    
    # Multi-objective weights
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "entanglement": 0.3,
        "novelty": 0.3,
        "depth_penalty": -0.1,
        "gate_count_penalty": -0.1,
        "hardware_fidelity": 0.2
    })
    
    # Novelty search parameters
    novelty_k_nearest: int = 15
    novelty_threshold: float = 0.5
    archive_size: int = 1000


class MutationOperator:
    """Quantum circuit mutation operators"""
    
    @staticmethod
    def insert_gate(genome: QuantumGenome, config: EvolutionConfig) -> QuantumGenome:
        """Insert a random gate at a random position"""
        mutant = genome.clone()
        
        if len(mutant.genes) >= config.max_gates:
            return mutant
            
        # Create random gene
        if config.native_gates_only:
            gate_types = [GateType.GPI, GateType.GPI2, GateType.MS]
        else:
            gate_types = [GateType.H, GateType.X, GateType.CNOT, GateType.RZ]
            
        gate_type = random.choice(gate_types)
        
        if gate_type == GateType.CNOT:
            if mutant.qubit_count < 2:
                return mutant
            q1, q2 = random.sample(range(mutant.qubit_count), 2)
            gene = Gene(gate_type, [q2], [q1])
        elif gate_type == GateType.MS:
            if mutant.qubit_count < 2:
                return mutant
            q1, q2 = random.sample(range(mutant.qubit_count), 2)
            phases = [random.uniform(0, 2*np.pi) for _ in range(2)]
            gene = Gene(gate_type, [q1, q2], parameters={"phases": phases})
        elif gate_type in [GateType.GPI, GateType.GPI2]:
            target = random.randint(0, mutant.qubit_count - 1)
            phase = random.uniform(0, 2*np.pi)
            gene = Gene(gate_type, [target], parameters={"phase": phase})
        elif gate_type == GateType.RZ:
            target = random.randint(0, mutant.qubit_count - 1)
            angle = random.uniform(0, 2*np.pi)
            gene = Gene(gate_type, [target], parameters={"angle": angle})
        else:
            target = random.randint(0, mutant.qubit_count - 1)
            gene = Gene(gate_type, [target])
            
        # Insert at random position
        position = random.randint(0, len(mutant.genes))
        mutant.genes.insert(position, gene)
        
        return mutant
    
    @staticmethod
    def delete_gate(genome: QuantumGenome, config: EvolutionConfig) -> QuantumGenome:
        """Delete a random gate"""
        mutant = genome.clone()
        
        if len(mutant.genes) <= 1:
            return mutant
            
        position = random.randint(0, len(mutant.genes) - 1)
        del mutant.genes[position]
        
        return mutant
    
    @staticmethod
    def modify_parameter(genome: QuantumGenome, config: EvolutionConfig) -> QuantumGenome:
        """Modify a parameter of a parametric gate"""
        mutant = genome.clone()
        
        # Find parametric gates
        parametric_indices = [
            i for i, gene in enumerate(mutant.genes)
            if gene.parameters is not None and len(gene.parameters) > 0
        ]
        
        if not parametric_indices:
            return mutant
            
        idx = random.choice(parametric_indices)
        gene = mutant.genes[idx]
        
        # Modify parameters
        if "angle" in gene.parameters:
            # Add noise to angle
            noise = np.random.normal(0, np.pi/4)
            gene.parameters["angle"] = (gene.parameters["angle"] + noise) % (2*np.pi)
        elif "phase" in gene.parameters:
            noise = np.random.normal(0, np.pi/4)
            gene.parameters["phase"] = (gene.parameters["phase"] + noise) % (2*np.pi)
        elif "phases" in gene.parameters:
            gene.parameters["phases"] = [
                (p + np.random.normal(0, np.pi/4)) % (2*np.pi) 
                for p in gene.parameters["phases"]
            ]
            
        return mutant
    
    @staticmethod
    def swap_gates(genome: QuantumGenome, config: EvolutionConfig) -> QuantumGenome:
        """Swap two random gates"""
        mutant = genome.clone()
        
        if len(mutant.genes) < 2:
            return mutant
            
        idx1, idx2 = random.sample(range(len(mutant.genes)), 2)
        mutant.genes[idx1], mutant.genes[idx2] = mutant.genes[idx2], mutant.genes[idx1]
        
        return mutant
    
    @staticmethod
    def change_target(genome: QuantumGenome, config: EvolutionConfig) -> QuantumGenome:
        """Change target qubit of a random gate"""
        mutant = genome.clone()
        
        if not mutant.genes or mutant.qubit_count < 2:
            return mutant
            
        idx = random.randint(0, len(mutant.genes) - 1)
        gene = mutant.genes[idx]
        
        # Change target qubit
        if gene.gate_type in [GateType.CNOT]:
            # For CNOT, ensure control != target
            new_target = random.randint(0, mutant.qubit_count - 1)
            while new_target in gene.controls:
                new_target = random.randint(0, mutant.qubit_count - 1)
            gene.targets = [new_target]
        elif gene.gate_type in [GateType.MS]:
            # For two-qubit gates, pick new pair
            q1, q2 = random.sample(range(mutant.qubit_count), 2)
            gene.targets = [q1, q2]
        else:
            # Single qubit gate
            gene.targets = [random.randint(0, mutant.qubit_count - 1)]
            
        return mutant


class CrossoverOperator:
    """Quantum circuit crossover operators"""
    
    @staticmethod
    def single_point_crossover(parent1: QuantumGenome, 
                              parent2: QuantumGenome) -> Tuple[QuantumGenome, QuantumGenome]:
        """Single point crossover"""
        if len(parent1.genes) < 2 or len(parent2.genes) < 2:
            return parent1.clone(), parent2.clone()
            
        point1 = random.randint(1, len(parent1.genes) - 1)
        point2 = random.randint(1, len(parent2.genes) - 1)
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        child1.genes = parent1.genes[:point1] + parent2.genes[point2:]
        child2.genes = parent2.genes[:point2] + parent1.genes[point1:]
        
        # Update parent tracking
        child1.parent_ids = [parent1.id, parent2.id]
        child2.parent_ids = [parent1.id, parent2.id]
        
        # Inherit qubit count from parent with more qubits
        max_qubits = max(parent1.qubit_count, parent2.qubit_count)
        child1.qubit_count = max_qubits
        child2.qubit_count = max_qubits
        
        return child1, child2
    
    @staticmethod
    def uniform_crossover(parent1: QuantumGenome, 
                         parent2: QuantumGenome) -> Tuple[QuantumGenome, QuantumGenome]:
        """Uniform crossover - each gene has 50% chance from each parent"""
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        max_len = max(len(parent1.genes), len(parent2.genes))
        
        child1.genes = []
        child2.genes = []
        
        for i in range(max_len):
            if random.random() < 0.5:
                if i < len(parent1.genes):
                    child1.genes.append(parent1.genes[i].clone())
                if i < len(parent2.genes):
                    child2.genes.append(parent2.genes[i].clone())
            else:
                if i < len(parent2.genes):
                    child1.genes.append(parent2.genes[i].clone())
                if i < len(parent1.genes):
                    child2.genes.append(parent1.genes[i].clone())
                    
        child1.parent_ids = [parent1.id, parent2.id]
        child2.parent_ids = [parent1.id, parent2.id]
        
        max_qubits = max(parent1.qubit_count, parent2.qubit_count)
        child1.qubit_count = max_qubits
        child2.qubit_count = max_qubits
        
        return child1, child2


class SelectionStrategy:
    """Selection strategies for evolutionary algorithm"""
    
    @staticmethod
    def tournament_selection(population: List[QuantumGenome],
                            fitness_scores: Dict[str, float],
                            tournament_size: int = 5) -> QuantumGenome:
        """Tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Get combined fitness for each individual
        def get_fitness(genome):
            return fitness_scores.get(genome.id, 0.0)
            
        winner = max(tournament, key=get_fitness)
        return winner
    
    @staticmethod
    def roulette_wheel_selection(population: List[QuantumGenome],
                                 fitness_scores: Dict[str, float]) -> QuantumGenome:
        """Roulette wheel selection"""
        # Normalize fitness scores
        total_fitness = sum(fitness_scores.values())
        if total_fitness == 0:
            return random.choice(population)
            
        probabilities = [fitness_scores.get(g.id, 0) / total_fitness 
                        for g in population]
        
        return np.random.choice(population, p=probabilities)
    
    @staticmethod
    def rank_selection(population: List[QuantumGenome],
                      fitness_scores: Dict[str, float]) -> QuantumGenome:
        """Rank-based selection"""
        # Sort by fitness
        sorted_pop = sorted(population, 
                           key=lambda g: fitness_scores.get(g.id, 0))
        
        # Assign rank-based probabilities
        ranks = np.arange(1, len(sorted_pop) + 1)
        probabilities = ranks / ranks.sum()
        
        return np.random.choice(sorted_pop, p=probabilities)


class NoveltyArchive:
    """Archive for novelty search"""
    
    def __init__(self, max_size: int = 1000):
        self.archive: List[QuantumGenome] = []
        self.max_size = max_size
        self.behavior_cache: Dict[str, np.ndarray] = {}
        
    def add(self, genome: QuantumGenome, behavior: np.ndarray):
        """Add genome to archive if novel enough"""
        self.behavior_cache[genome.id] = behavior
        
        if len(self.archive) < self.max_size:
            self.archive.append(genome)
        else:
            # Replace least novel individual
            novelties = [self.compute_novelty(g) for g in self.archive]
            min_idx = np.argmin(novelties)
            if self.compute_novelty(genome) > novelties[min_idx]:
                self.archive[min_idx] = genome
                
    def compute_novelty(self, genome: QuantumGenome, k: int = 15) -> float:
        """Compute novelty score based on k-nearest neighbors"""
        if genome.id not in self.behavior_cache:
            return 0.0
            
        behavior = self.behavior_cache[genome.id]
        
        # Compute distances to all archived behaviors
        # Limit comparisons if archive is too large to prevent hanging
        max_comparisons = 500
        archive_to_compare = self.archive
        if len(self.archive) > max_comparisons:
            # Randomly sample a subset for comparison
            indices = np.random.choice(len(self.archive), size=max_comparisons, replace=False)
            archive_to_compare = [self.archive[i] for i in indices]
        
        distances = []
        for other in archive_to_compare:
            if other.id != genome.id and other.id in self.behavior_cache:
                other_behavior = self.behavior_cache[other.id]
                dist = np.linalg.norm(behavior - other_behavior)
                distances.append(dist)
                
        if not distances:
            return float('inf')  # First individual is maximally novel
            
        # Average distance to k nearest neighbors
        k = min(k, len(distances))
        nearest_distances = sorted(distances)[:k]
        
        return np.mean(nearest_distances)
    
    def get_diverse_subset(self, n: int) -> List[QuantumGenome]:
        """Get n most diverse genomes from archive"""
        if len(self.archive) <= n:
            return self.archive.copy()
            
        # Use k-means clustering to find diverse subset
        behaviors = np.array([
            self.behavior_cache.get(g.id, np.zeros(10))
            for g in self.archive
        ])
        
        # Simple diversity sampling
        selected = []
        remaining = self.archive.copy()
        
        # Start with random genome
        first = random.choice(remaining)
        selected.append(first)
        remaining.remove(first)
        
        # Greedily select most distant genomes
        while len(selected) < n and remaining:
            max_dist = -1
            max_genome = None
            
            for genome in remaining:
                if genome.id not in self.behavior_cache:
                    continue
                    
                behavior = self.behavior_cache[genome.id]
                
                # Min distance to selected genomes
                min_dist_to_selected = float('inf')
                for sel in selected:
                    if sel.id in self.behavior_cache:
                        sel_behavior = self.behavior_cache[sel.id]
                        dist = np.linalg.norm(behavior - sel_behavior)
                        min_dist_to_selected = min(min_dist_to_selected, dist)
                        
                if min_dist_to_selected > max_dist:
                    max_dist = min_dist_to_selected
                    max_genome = genome
                    
            if max_genome:
                selected.append(max_genome)
                remaining.remove(max_genome)
            else:
                break
                
        return selected


class EvolutionEngine:
    """Main evolutionary algorithm engine"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.population: List[QuantumGenome] = []
        self.generation = 0
        self.novelty_archive = NoveltyArchive(config.archive_size)
        self.fitness_history: List[Dict[str, float]] = []
        self.elite_hall_of_fame: List[QuantumGenome] = []
        
        # Mutation operators with probabilities
        self.mutation_operators = [
            (0.3, MutationOperator.insert_gate),
            (0.2, MutationOperator.delete_gate),
            (0.2, MutationOperator.modify_parameter),
            (0.15, MutationOperator.swap_gates),
            (0.15, MutationOperator.change_target),
        ]
        
    def initialize_population(self):
        """Initialize population with diverse circuits"""
        self.population = []
        
        # Add known good circuits (20%)
        known_circuits = int(self.config.population_size * 0.2)
        for _ in range(known_circuits // 3):
            self.population.append(GenomeFactory.create_bell_pair())
        for _ in range(known_circuits // 3):
            self.population.append(GenomeFactory.create_ghz(3))
        for _ in range(known_circuits // 3):
            self.population.append(GenomeFactory.create_qft(3))
            
        # Add random circuits (80%)
        while len(self.population) < self.config.population_size:
            genome = GenomeFactory.create_random(
                qubit_count=random.randint(2, self.config.qubit_budget),
                max_gates=self.config.max_gates,
                native_only=self.config.native_gates_only
            )
            self.population.append(genome)
            
        # Set generation for all
        for genome in self.population:
            genome.generation = 0
            
    def mutate(self, genome: QuantumGenome) -> QuantumGenome:
        """Apply mutation to genome"""
        if random.random() > self.config.mutation_rate:
            return genome
            
        # Select mutation operator based on probabilities
        r = random.random()
        cumulative = 0
        
        for prob, operator in self.mutation_operators:
            cumulative += prob
            if r < cumulative:
                return operator(genome, self.config)
                
        return genome
    
    def crossover(self, parent1: QuantumGenome, 
                 parent2: QuantumGenome) -> Tuple[QuantumGenome, QuantumGenome]:
        """Apply crossover to create offspring"""
        if random.random() > self.config.crossover_rate:
            return parent1.clone(), parent2.clone()
            
        if random.random() < 0.5:
            return CrossoverOperator.single_point_crossover(parent1, parent2)
        else:
            return CrossoverOperator.uniform_crossover(parent1, parent2)
    
    def select_parents(self, fitness_scores: Dict[str, float]) -> Tuple[QuantumGenome, QuantumGenome]:
        """Select two parents for reproduction"""
        parent1 = SelectionStrategy.tournament_selection(
            self.population, fitness_scores, self.config.tournament_size
        )
        parent2 = SelectionStrategy.tournament_selection(
            self.population, fitness_scores, self.config.tournament_size
        )
        
        # Ensure different parents
        while parent2.id == parent1.id and len(self.population) > 1:
            parent2 = SelectionStrategy.tournament_selection(
                self.population, fitness_scores, self.config.tournament_size
            )
            
        return parent1, parent2
    
    def evolve_generation(self, fitness_evaluator: Callable) -> Dict[str, float]:
        """Evolve one generation"""
        # Evaluate current population
        fitness_scores = {}
        behaviors = {}
        
        for genome in self.population:
            fitness, behavior = fitness_evaluator(genome)
            fitness_scores[genome.id] = fitness
            behaviors[genome.id] = behavior
            genome.fitness_scores = {"combined": fitness}
            genome.behavior_descriptor = behavior
            
            # Add to novelty archive
            self.novelty_archive.add(genome, behavior)
            
        # Update novelty scores
        for genome in self.population:
            genome.novelty_score = self.novelty_archive.compute_novelty(
                genome, self.config.novelty_k_nearest
            )
            
        # Select elites
        sorted_pop = sorted(self.population, 
                           key=lambda g: fitness_scores[g.id], 
                           reverse=True)
        elites = sorted_pop[:self.config.elite_size]
        
        # Update hall of fame
        self.elite_hall_of_fame.extend(elites[:3])
        self.elite_hall_of_fame = sorted(
            self.elite_hall_of_fame,
            key=lambda g: fitness_scores.get(g.id, 0),
            reverse=True
        )[:100]  # Keep top 100 all-time
        
        # Create next generation
        next_generation = [e.clone() for e in elites]  # Keep elites
        
        while len(next_generation) < self.config.population_size:
            # Select parents
            parent1, parent2 = self.select_parents(fitness_scores)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Update generation
            child1.generation = self.generation + 1
            child2.generation = self.generation + 1
            
            # Validate and add
            if child1.validate():
                next_generation.append(child1)
            if len(next_generation) < self.config.population_size and child2.validate():
                next_generation.append(child2)
                
        self.population = next_generation[:self.config.population_size]
        self.generation += 1
        
        # Store fitness history
        # Convert behavior arrays to tuples for hashing
        unique_behaviors = set(tuple(b.tolist()) if isinstance(b, np.ndarray) else b 
                              for b in behaviors.values())
        self.fitness_history.append({
            "generation": self.generation,
            "best": max(fitness_scores.values()),
            "average": np.mean(list(fitness_scores.values())),
            "diversity": len(unique_behaviors)
        })
        
        return fitness_scores
    
    def get_statistics(self) -> Dict:
        """Get current evolution statistics"""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "archive_size": len(self.novelty_archive.archive),
            "hall_of_fame_size": len(self.elite_hall_of_fame),
            "fitness_history": self.fitness_history[-10:] if self.fitness_history else []
        }