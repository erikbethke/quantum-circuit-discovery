"""
DFAL - Discover First, Apply Later
Quantum Circuit Discovery Engine
"""

__version__ = "0.1.0"
__author__ = "Erik Bethke"
__email__ = "erik@bike4mind.com"

from .core.genome import QuantumGenome, GenomeFactory
from .core.evolution import EvolutionEngine, EvolutionConfig
from .core.fitness import FitnessEvaluator, FitnessMetrics

__all__ = [
    "QuantumGenome",
    "GenomeFactory", 
    "EvolutionEngine",
    "EvolutionConfig",
    "FitnessEvaluator",
    "FitnessMetrics"
]