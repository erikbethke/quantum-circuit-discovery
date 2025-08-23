#!/usr/bin/env python3
"""
DFAL - Discover First, Apply Later
Main entry point for quantum circuit discovery engine
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core modules
from src.core.genome import QuantumGenome, GenomeFactory
from src.core.evolution import EvolutionEngine, EvolutionConfig
from src.core.fitness import FitnessEvaluator
from src.ionq.client import IonQClient, IonQConfig, IonQSimulator, IonQBackend


class DFALEngine:
    """Main DFAL discovery engine"""
    
    def __init__(self, 
                 use_hardware: bool = False,
                 output_dir: str = "discoveries"):
        self.use_hardware = use_hardware
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.evolution_config = EvolutionConfig(
            population_size=50,
            elite_size=5,
            mutation_rate=0.3,
            crossover_rate=0.7,
            max_circuit_depth=20,
            qubit_budget=4,
            native_gates_only=use_hardware
        )
        
        self.evolution_engine = EvolutionEngine(self.evolution_config)
        self.fitness_evaluator = FitnessEvaluator()
        
        # IonQ setup
        if use_hardware:
            self.ionq_config = IonQConfig()
            self.ionq_client = None  # Will initialize in async context
        else:
            self.simulator = IonQSimulator()
            
        # Discovery tracking
        self.discoveries = []
        self.generation_stats = []
        
    async def initialize(self):
        """Initialize async components"""
        if self.use_hardware:
            self.ionq_client = IonQClient(self.ionq_config)
            await self.ionq_client.__aenter__()
            
    async def cleanup(self):
        """Cleanup async components"""
        if self.use_hardware and self.ionq_client:
            await self.ionq_client.__aexit__(None, None, None)
            
    def evaluate_genome(self, genome: QuantumGenome) -> tuple:
        """Evaluate a single genome"""
        # Simulate locally for now
        if not self.use_hardware:
            sim_result = self.simulator.simulate(genome, shots=1000)
        else:
            # Would use IonQ here in production
            sim_result = {"probabilities": {}, "warning": "Hardware not implemented yet"}
            
        # Compute fitness
        fitness, behavior = self.fitness_evaluator.evaluate(genome, sim_result)
        
        return fitness, behavior
    
    async def run_evolution(self, generations: int = 10):
        """Run evolutionary discovery"""
        logger.info(f"Starting evolution for {generations} generations")
        
        # Initialize population
        self.evolution_engine.initialize_population()
        logger.info(f"Initialized population of {len(self.evolution_engine.population)} circuits")
        
        for gen in range(generations):
            logger.info(f"\n{'='*50}")
            logger.info(f"Generation {gen + 1}/{generations}")
            logger.info(f"{'='*50}")
            
            # Evolve one generation
            fitness_scores = self.evolution_engine.evolve_generation(
                self.evaluate_genome
            )
            
            # Get statistics
            stats = self.evolution_engine.get_statistics()
            self.generation_stats.append(stats)
            
            # Log progress
            best_fitness = max(fitness_scores.values())
            avg_fitness = np.mean(list(fitness_scores.values()))
            
            logger.info(f"Best fitness: {best_fitness:.4f}")
            logger.info(f"Average fitness: {avg_fitness:.4f}")
            logger.info(f"Archive size: {stats['archive_size']}")
            logger.info(f"Hall of fame size: {stats['hall_of_fame_size']}")
            
            # Check for discoveries
            for genome in self.evolution_engine.population:
                if genome.fitness_scores.get("combined", 0) > 0.7:
                    self.record_discovery(genome, gen + 1)
                    
            # Save checkpoint
            if (gen + 1) % 5 == 0:
                self.save_checkpoint(gen + 1)
                
        logger.info("\nEvolution complete!")
        self.save_hall_of_fame()
        self.print_summary()
        
    def record_discovery(self, genome: QuantumGenome, generation: int):
        """Record a significant discovery"""
        discovery = {
            "id": genome.id,
            "generation": generation,
            "fitness": genome.fitness_scores.get("combined", 0),
            "depth": genome.depth(),
            "gates": len(genome.genes),
            "qubits": genome.qubit_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if already discovered
        if not any(d["id"] == genome.id for d in self.discoveries):
            self.discoveries.append(discovery)
            logger.info(f"🎯 DISCOVERY: Circuit {genome.id[:8]} with fitness {discovery['fitness']:.4f}")
            
            # Save circuit
            self.save_circuit(genome)
            
    def save_circuit(self, genome: QuantumGenome):
        """Save discovered circuit"""
        circuit_file = self.output_dir / f"circuit_{genome.id[:8]}.json"
        
        circuit_data = {
            "id": genome.id,
            "generation": genome.generation,
            "fitness": genome.fitness_scores,
            "metrics": {
                "depth": genome.depth(),
                "gates": len(genome.genes),
                "two_qubit_gates": genome.two_qubit_gate_count(),
                "qubits": genome.qubit_count
            },
            "circuit": genome.to_circuit_json(),
            "genes": [
                {
                    "gate": g.gate_type.value,
                    "targets": g.targets,
                    "controls": g.controls,
                    "parameters": g.parameters
                }
                for g in genome.genes
            ],
            "discovery_notes": genome.discovery_notes,
            "classification": genome.classification
        }
        
        with open(circuit_file, 'w') as f:
            json.dump(circuit_data, f, indent=2)
            
    def save_hall_of_fame(self):
        """Save all Hall of Fame circuits"""
        logger.info(f"Saving {len(self.evolution_engine.elite_hall_of_fame)} Hall of Fame circuits...")
        
        for i, genome in enumerate(self.evolution_engine.elite_hall_of_fame):
            circuit_file = self.output_dir / f"hall_of_fame_{i+1}_{genome.id[:8]}.json"
            
            circuit_data = {
                "id": genome.id,
                "rank": i + 1,
                "fitness": genome.fitness_scores,
                "metadata": {
                    "depth": genome.depth(),
                    "gate_count": len(genome.genes),
                    "qubit_count": genome.qubit_count,
                    "unique_gates": list({g.gate_type.value for g in genome.genes})
                },
                "circuit": {
                    "qubits": genome.qubit_count,
                    "gates": [
                        {
                            "gate": gene.gate_type.value,
                            "targets": gene.targets,
                            "controls": gene.controls,
                            "parameters": gene.parameters
                        }
                        for gene in genome.genes
                    ]
                }
            }
            
            with open(circuit_file, 'w') as f:
                json.dump(circuit_data, f, indent=2)
                
        logger.info(f"Hall of Fame circuits saved to {self.output_dir}")
    
    def save_checkpoint(self, generation: int):
        """Save evolution checkpoint"""
        checkpoint_file = self.output_dir / f"checkpoint_gen{generation}.json"
        
        checkpoint = {
            "generation": generation,
            "config": {
                "population_size": self.evolution_config.population_size,
                "mutation_rate": self.evolution_config.mutation_rate,
                "crossover_rate": self.evolution_config.crossover_rate
            },
            "statistics": self.generation_stats[-5:],  # Last 5 generations
            "discoveries": self.discoveries,
            "top_circuits": []
        }
        
        # Save top 10 circuits
        top_circuits = sorted(
            self.evolution_engine.elite_hall_of_fame[:10],
            key=lambda g: g.fitness_scores.get("combined", 0),
            reverse=True
        )
        
        for genome in top_circuits:
            checkpoint["top_circuits"].append({
                "id": genome.id,
                "fitness": genome.fitness_scores.get("combined", 0),
                "depth": genome.depth(),
                "gates": len(genome.genes)
            })
            
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
        logger.info(f"Saved checkpoint at generation {generation}")
        
    def print_summary(self):
        """Print evolution summary"""
        print("\n" + "="*60)
        print("EVOLUTION SUMMARY")
        print("="*60)
        
        print(f"\nTotal discoveries: {len(self.discoveries)}")
        
        if self.discoveries:
            print("\nTop 5 Discoveries:")
            top_discoveries = sorted(
                self.discoveries, 
                key=lambda d: d["fitness"], 
                reverse=True
            )[:5]
            
            for i, disc in enumerate(top_discoveries, 1):
                print(f"{i}. Circuit {disc['id'][:8]}")
                print(f"   Fitness: {disc['fitness']:.4f}")
                print(f"   Generation: {disc['generation']}")
                print(f"   Depth: {disc['depth']}, Gates: {disc['gates']}, Qubits: {disc['qubits']}")
                
        print("\nHall of Fame:")
        for i, genome in enumerate(self.evolution_engine.elite_hall_of_fame[:5], 1):
            print(f"{i}. {genome.id[:8]} - Fitness: {genome.fitness_scores.get('combined', 0):.4f}")
            
        print(f"\nOutput saved to: {self.output_dir}")
        print("="*60)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DFAL Quantum Circuit Discovery Engine")
    parser.add_argument(
        "--generations", 
        type=int, 
        default=10,
        help="Number of generations to evolve"
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Use IonQ hardware instead of simulator"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="discoveries",
        help="Output directory for discoveries"
    )
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     DFAL - Discover First, Apply Later                      ║
    ║     Quantum Circuit Discovery Engine v0.1                   ║
    ║                                                              ║
    ║     🧬 Evolving quantum circuits beyond human imagination   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create engine
    engine = DFALEngine(
        use_hardware=args.hardware,
        output_dir=args.output
    )
    
    try:
        # Initialize
        await engine.initialize()
        
        # Run evolution
        await engine.run_evolution(generations=args.generations)
        
    finally:
        # Cleanup
        await engine.cleanup()
        

if __name__ == "__main__":
    asyncio.run(main())