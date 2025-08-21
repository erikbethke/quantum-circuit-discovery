# 🚀 DFAL - Discover First, Apply Later

## Quantum Circuit Discovery Engine

An evolutionary AI system that discovers novel quantum circuits and retroactively finds their applications. Instead of designing circuits for specific problems, DFAL explores the vast space of quantum circuits to find hidden gems that humans would never conceive.

## 🎯 Vision

**"What if we let machines explore the 10 million possible quantum circuits on today's hardware and discover the algorithms we haven't even imagined yet?"**

This system implements the groundbreaking "Discover First, Apply Later" paradigm - using evolutionary algorithms, novelty search, and AI to:
1. Generate millions of quantum circuit candidates
2. Evolve them toward interesting quantum behaviors 
3. Use AI to understand what they do
4. Match them to real-world applications

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DFAL Discovery Engine                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Evolution   │  │   Fitness    │  │    IonQ      │  │
│  │   Engine     │──│  Evaluator   │──│  Integration │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                 │                   │          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Novelty    │  │      AI      │  │  Application │  │
│  │   Archive    │  │  Classifier  │  │    Mapper    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone [repository-url]
cd quantumLab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your IONQ_API_KEY
```

### 2. Run Discovery Engine

```bash
# Run with simulator (no API key needed)
python main.py --generations 20

# Run with IonQ hardware (requires API key)
python main.py --generations 10 --hardware

# Custom output directory
python main.py --generations 50 --output my_discoveries
```

### 3. Monitor Progress

The system will output:
- Real-time evolution statistics
- Discovery notifications when interesting circuits are found
- Saved circuits in JSON format
- Checkpoints every 5 generations

## 📊 Example Output

```
Generation 10/20
==================================================
Best fitness: 0.8234
Average fitness: 0.5671
Archive size: 127
Hall of fame size: 23

🎯 DISCOVERY: Circuit a3f2b1c8 with fitness 0.8234
   - 4 qubits, depth 12, 18 gates
   - High entanglement (0.92)
   - Novel behavior pattern
```

## 🧬 Core Features

### Evolutionary Engine
- **Population-based search** with configurable size
- **Multi-objective optimization** (entanglement, novelty, efficiency)
- **Adaptive mutation operators** specific to quantum circuits
- **Elite preservation** and hall of fame tracking

### Quantum Genome Representation
- **Flexible encoding** supporting any gate set
- **IonQ native gate** support (GPI, GPI2, MS)
- **Hierarchical genes** - can evolve high-level primitives
- **Automatic validation** and constraint checking

### Fitness Evaluation
- **Meyer-Wallach entanglement** measure
- **Circuit depth and gate efficiency** 
- **Hardware fidelity estimation** for IonQ devices
- **Novelty scoring** based on behavioral uniqueness
- **Output complexity** via Shannon entropy

### Novelty Search & MAP-Elites
- **Quality-Diversity archive** maintaining diverse solutions
- **Behavioral descriptors** capturing circuit characteristics
- **K-nearest neighbor novelty** computation
- **Automatic archiving** of interesting discoveries

## 🔬 Advanced Usage

### Custom Fitness Weights

```python
from src.core.evolution import EvolutionConfig

config = EvolutionConfig(
    fitness_weights={
        "entanglement": 0.4,
        "novelty": 0.4,
        "hardware_fidelity": 0.2
    }
)
```

### Seed with Known Circuits

```python
from src.core.genome import GenomeFactory

# Add to initial population
bell_pair = GenomeFactory.create_bell_pair()
ghz_state = GenomeFactory.create_ghz(4)
qft_circuit = GenomeFactory.create_qft(3)
```

### Batch Hardware Execution

```python
from src.ionq.client import IonQBatchProcessor

processor = IonQBatchProcessor(client)
results = await processor.process_generation(
    genomes, 
    backend=IonQBackend.ARIA_1,
    shots=1000
)
```

## 📈 Discoveries Format

Each discovered circuit is saved as JSON:

```json
{
  "id": "a3f2b1c8-...",
  "generation": 10,
  "fitness": {
    "combined": 0.8234,
    "entanglement": 0.92,
    "novelty": 0.78
  },
  "metrics": {
    "depth": 12,
    "gates": 18,
    "two_qubit_gates": 6,
    "qubits": 4
  },
  "circuit": {
    "format": "ionq.circuit.v0",
    "qubits": 4,
    "circuit": [...]
  },
  "classification": "Potential quantum walk variant",
  "discovery_notes": "High entanglement with shallow depth"
}
```

## 🏆 Patent Portfolio

This implementation supports three provisional patents:

1. **Evolutionary Quantum Circuit Generator** - The core discovery engine
2. **Agentic AI Evaluation & Classification** - AI-powered circuit understanding  
3. **Post-Hoc Application Mapping** - Matching circuits to applications

## 🤝 Contributing

This is a research project pushing the boundaries of quantum computing. Contributions welcome in:

- Novel fitness metrics
- Better behavioral descriptors
- Improved mutation operators
- Application mapping strategies
- Visualization tools

## 📚 References

- Spector, L. "Machine Invention of Quantum Computing Circuits by Means of GP" (2008)
- Krenn, M. et al. "Automated Search for New Quantum Experiments" (2016)
- YAQQ: Novelty Search for Quantum Gate Sets (2024)

## ⚡ Performance

On a modern laptop:
- **Population 100**: ~30 seconds per generation
- **Population 500**: ~3 minutes per generation
- **With IonQ API**: +2-5 seconds per circuit

## 🔮 Future Roadmap

- [ ] Distributed evolution across multiple machines
- [ ] Real-time visualization dashboard
- [ ] Integration with more quantum backends
- [ ] Automated scientific paper generation for discoveries
- [ ] Reinforcement learning for guided exploration

## 📞 Support

For IonQ partnership inquiries: [Contact Info]
For technical questions: [Create an issue]

---

**"The best quantum algorithm might already exist in the space of all possible circuits. We just need to find it."**

Built with ❤️ for the quantum future by Erik Bethke & Bike4Mind