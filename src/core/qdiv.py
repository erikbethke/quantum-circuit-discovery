"""
Quality-Diversity Archive (MAP-Elites) for Quantum Circuits
Maintains diverse collection of high-performing circuits
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import pickle
from pathlib import Path
from datetime import datetime

from ..core.genome import QuantumGenome


@dataclass
class ArchiveCell:
    """Single cell in MAP-Elites archive"""
    genome: QuantumGenome
    fitness: float
    behavior: np.ndarray
    metrics: Dict[str, float]
    generation: int
    timestamp: datetime = field(default_factory=datetime.now)
    

@dataclass 
class QDArchiveConfig:
    """Configuration for MAP-Elites archive"""
    # Behavior space dimensions
    bins: Dict[str, List[float]] = field(default_factory=lambda: {
        'entanglement': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'depth': [0, 5, 10, 15, 20, 30],
        'fidelity': [0.0, 0.5, 0.7, 0.85, 0.95, 1.0]
    })
    
    # Archive management
    max_archive_size: int = 10000
    replacement_strategy: str = 'elite'  # 'elite', 'probabilistic', 'age'
    age_weight: float = 0.1  # For age-based replacement
    
    # Diversity pressure
    novelty_weight: float = 0.3
    local_competition: bool = True
    
    # Persistence
    save_frequency: int = 10  # Save every N generations
    save_path: str = 'archives'


class MAPElites:
    """
    MAP-Elites implementation for quantum circuit discovery
    Maintains grid of diverse, high-quality solutions
    """
    
    def __init__(self, config: Optional[QDArchiveConfig] = None):
        self.config = config or QDArchiveConfig()
        
        # Initialize grid
        self.grid: Dict[Tuple, ArchiveCell] = {}
        self.dimension_names = list(self.config.bins.keys())
        self.n_dims = len(self.dimension_names)
        
        # Compute total cells
        self.grid_shape = tuple(len(bins) - 1 for bins in self.config.bins.values())
        self.total_cells = np.prod(self.grid_shape)
        
        # Statistics
        self.stats = {
            'total_evaluations': 0,
            'total_insertions': 0,
            'coverage': 0.0,
            'best_fitness': 0.0,
            'generation': 0
        }
        
        # History for analysis
        self.history = []
        
    def compute_cell_key(self, metrics: Dict[str, float]) -> Optional[Tuple[int, ...]]:
        """
        Map metrics to grid cell coordinates
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            Tuple of bin indices, or None if out of bounds
        """
        indices = []
        
        for dim_name in self.dimension_names:
            if dim_name not in metrics:
                return None
                
            value = metrics[dim_name]
            bins = self.config.bins[dim_name]
            
            # Find bin index
            bin_idx = np.searchsorted(bins[1:], value)
            
            # Check bounds
            if bin_idx >= len(bins) - 1:
                bin_idx = len(bins) - 2
                
            indices.append(bin_idx)
            
        return tuple(indices)
    
    def insert(self, genome: QuantumGenome, 
              fitness: float,
              metrics: Dict[str, float],
              behavior: np.ndarray,
              generation: int) -> bool:
        """
        Try to insert genome into archive
        
        Args:
            genome: Quantum circuit genome
            fitness: Combined fitness score
            metrics: Behavior metrics
            behavior: Behavior descriptor vector
            generation: Current generation number
            
        Returns:
            True if inserted, False otherwise
        """
        self.stats['total_evaluations'] += 1
        
        # Compute cell coordinates
        cell_key = self.compute_cell_key(metrics)
        if cell_key is None:
            return False
            
        # Check if cell is empty or should be replaced
        should_insert = False
        
        if cell_key not in self.grid:
            # Empty cell - always insert
            should_insert = True
        else:
            # Cell occupied - check replacement
            current = self.grid[cell_key]
            
            if self.config.replacement_strategy == 'elite':
                # Replace if better fitness
                should_insert = fitness > current.fitness
                
            elif self.config.replacement_strategy == 'probabilistic':
                # Probabilistic replacement based on fitness difference
                if fitness > current.fitness:
                    should_insert = True
                else:
                    prob = np.exp((fitness - current.fitness) / 0.1)
                    should_insert = np.random.random() < prob
                    
            elif self.config.replacement_strategy == 'age':
                # Consider both fitness and age
                age_factor = (generation - current.generation) * self.config.age_weight
                should_insert = (fitness + age_factor) > current.fitness
                
        if should_insert:
            # Create archive cell
            cell = ArchiveCell(
                genome=genome.clone(),
                fitness=fitness,
                behavior=behavior.copy(),
                metrics=metrics.copy(),
                generation=generation
            )
            
            self.grid[cell_key] = cell
            self.stats['total_insertions'] += 1
            
            # Update best fitness
            if fitness > self.stats['best_fitness']:
                self.stats['best_fitness'] = fitness
                
            return True
            
        return False
    
    def get_random_elites(self, n: int) -> List[QuantumGenome]:
        """
        Sample random elites from archive
        
        Args:
            n: Number of elites to sample
            
        Returns:
            List of genome copies
        """
        if len(self.grid) == 0:
            return []
            
        n = min(n, len(self.grid))
        cells = np.random.choice(list(self.grid.values()), n, replace=False)
        
        return [cell.genome.clone() for cell in cells]
    
    def get_diverse_elites(self, n: int) -> List[QuantumGenome]:
        """
        Get diverse set of elites using max-min selection
        
        Args:
            n: Number of elites to get
            
        Returns:
            List of diverse genomes
        """
        if len(self.grid) == 0:
            return []
            
        cells = list(self.grid.values())
        if len(cells) <= n:
            return [cell.genome.clone() for cell in cells]
            
        # Max-min selection on behavior descriptors
        selected_indices = [np.random.randint(len(cells))]
        
        while len(selected_indices) < n:
            max_min_dist = -1
            max_idx = None
            
            for i, cell in enumerate(cells):
                if i in selected_indices:
                    continue
                    
                # Min distance to selected cells
                min_dist = min(
                    np.linalg.norm(cell.behavior - cells[j].behavior)
                    for j in selected_indices
                )
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    max_idx = i
                    
            if max_idx is not None:
                selected_indices.append(max_idx)
            else:
                break
                
        return [cells[i].genome.clone() for i in selected_indices]
    
    def get_cell_neighbors(self, cell_key: Tuple[int, ...], 
                          radius: int = 1) -> List[Tuple[int, ...]]:
        """
        Get neighboring cells within radius
        
        Args:
            cell_key: Center cell coordinates
            radius: Neighborhood radius
            
        Returns:
            List of neighboring cell keys
        """
        neighbors = []
        
        # Generate all offset combinations
        for offset in np.ndindex(*([2*radius + 1] * self.n_dims)):
            neighbor_key = tuple(
                cell_key[i] + offset[i] - radius
                for i in range(self.n_dims)
            )
            
            # Check bounds
            valid = all(
                0 <= neighbor_key[i] < self.grid_shape[i]
                for i in range(self.n_dims)
            )
            
            if valid and neighbor_key != cell_key:
                neighbors.append(neighbor_key)
                
        return neighbors
    
    def local_competition(self, genome: QuantumGenome,
                         fitness: float,
                         metrics: Dict[str, float],
                         behavior: np.ndarray,
                         generation: int) -> bool:
        """
        Insert with local competition (compete with neighbors)
        
        Args:
            genome: Genome to insert
            fitness: Fitness score
            metrics: Behavior metrics
            behavior: Behavior descriptor
            generation: Current generation
            
        Returns:
            True if inserted
        """
        cell_key = self.compute_cell_key(metrics)
        if cell_key is None:
            return False
            
        # Get neighboring cells
        neighbors = self.get_cell_neighbors(cell_key, radius=1)
        
        # Include target cell
        competition_cells = [cell_key] + neighbors
        
        # Find worst performer in neighborhood
        worst_fitness = fitness
        worst_cell = cell_key
        
        for key in competition_cells:
            if key in self.grid:
                if self.grid[key].fitness < worst_fitness:
                    worst_fitness = self.grid[key].fitness
                    worst_cell = key
            else:
                # Empty cell - use it
                worst_cell = key
                worst_fitness = -float('inf')
                break
                
        # Insert into worst cell if better
        if fitness > worst_fitness:
            cell = ArchiveCell(
                genome=genome.clone(),
                fitness=fitness,
                behavior=behavior.copy(),
                metrics=metrics.copy(),
                generation=generation
            )
            
            self.grid[worst_cell] = cell
            self.stats['total_insertions'] += 1
            
            if fitness > self.stats['best_fitness']:
                self.stats['best_fitness'] = fitness
                
            return True
            
        return False
    
    def compute_coverage(self) -> float:
        """Compute archive coverage (filled cells / total cells)"""
        coverage = len(self.grid) / self.total_cells if self.total_cells > 0 else 0
        self.stats['coverage'] = coverage
        return coverage
    
    def compute_qd_score(self) -> float:
        """Compute Quality-Diversity score (sum of fitness in archive)"""
        if len(self.grid) == 0:
            return 0.0
        return sum(cell.fitness for cell in self.grid.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get archive statistics"""
        self.compute_coverage()
        
        stats = self.stats.copy()
        stats['n_filled'] = len(self.grid)
        stats['qd_score'] = self.compute_qd_score()
        
        if len(self.grid) > 0:
            fitnesses = [cell.fitness for cell in self.grid.values()]
            stats['mean_fitness'] = np.mean(fitnesses)
            stats['std_fitness'] = np.std(fitnesses)
            stats['min_fitness'] = np.min(fitnesses)
            stats['max_fitness'] = np.max(fitnesses)
            
            # Dimension coverage
            for dim_name in self.dimension_names:
                dim_values = [cell.metrics[dim_name] 
                             for cell in self.grid.values()]
                stats[f'{dim_name}_mean'] = np.mean(dim_values)
                stats[f'{dim_name}_std'] = np.std(dim_values)
                
        return stats
    
    def save(self, filepath: Optional[str] = None):
        """Save archive to disk"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(self.config.save_path) / f"archive_{timestamp}.pkl"
        else:
            filepath = Path(filepath)
            
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'grid': self.grid,
                'config': self.config,
                'stats': self.stats,
                'history': self.history
            }, f)
            
        # Also save readable summary
        summary_path = filepath.with_suffix('.json')
        summary = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_statistics(),
            'config': {
                'bins': self.config.bins,
                'total_cells': self.total_cells,
                'replacement_strategy': self.config.replacement_strategy
            },
            'top_genomes': []
        }
        
        # Add top 10 genomes
        sorted_cells = sorted(self.grid.values(), 
                            key=lambda c: c.fitness, 
                            reverse=True)[:10]
        
        for cell in sorted_cells:
            summary['top_genomes'].append({
                'id': cell.genome.id,
                'fitness': cell.fitness,
                'generation': cell.generation,
                'metrics': cell.metrics,
                'depth': cell.genome.depth(),
                'gates': len(cell.genome.genes)
            })
            
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return filepath
    
    def load(self, filepath: str):
        """Load archive from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.grid = data['grid']
        self.config = data['config']
        self.stats = data['stats']
        self.history = data.get('history', [])
        
        # Recompute derived values
        self.dimension_names = list(self.config.bins.keys())
        self.n_dims = len(self.dimension_names)
        self.grid_shape = tuple(len(bins) - 1 
                               for bins in self.config.bins.values())
        self.total_cells = np.prod(self.grid_shape)
    
    def visualize_grid(self) -> np.ndarray:
        """
        Create 2D visualization of archive (first 2 dimensions)
        
        Returns:
            2D array with fitness values
        """
        if self.n_dims < 2:
            return np.array([])
            
        grid_vis = np.zeros(self.grid_shape[:2])
        
        for cell_key, cell in self.grid.items():
            if len(cell_key) >= 2:
                grid_vis[cell_key[0], cell_key[1]] = cell.fitness
                
        return grid_vis
    
    def get_pareto_front(self) -> List[ArchiveCell]:
        """
        Get Pareto-optimal solutions from archive
        
        Returns:
            List of non-dominated cells
        """
        cells = list(self.grid.values())
        if len(cells) == 0:
            return []
            
        pareto_front = []
        
        for i, cell in enumerate(cells):
            dominated = False
            
            for j, other in enumerate(cells):
                if i == j:
                    continue
                    
                # Check if other dominates cell
                better_in_all = all(
                    other.metrics.get(dim, 0) >= cell.metrics.get(dim, 0)
                    for dim in self.dimension_names
                )
                
                better_in_one = any(
                    other.metrics.get(dim, 0) > cell.metrics.get(dim, 0)
                    for dim in self.dimension_names
                )
                
                if better_in_all and better_in_one:
                    dominated = True
                    break
                    
            if not dominated:
                pareto_front.append(cell)
                
        return pareto_front