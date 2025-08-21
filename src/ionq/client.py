"""
IonQ API Client for Quantum Circuit Execution
Handles job submission, monitoring, and result retrieval
"""

import os
import json
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

from ..core.genome import QuantumGenome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IonQBackend(Enum):
    """Available IonQ backends"""
    SIMULATOR = "simulator"
    ARIA_1 = "aria-1" 
    ARIA_2 = "aria-2"
    FORTE_1 = "forte-1"
    QPU = "qpu"  # Let IonQ choose


class JobStatus(Enum):
    """IonQ job status"""
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class IonQJob:
    """IonQ job representation"""
    id: str
    status: JobStatus
    backend: str
    shots: int
    created_at: datetime
    circuit: Dict
    results: Optional[Dict] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    

@dataclass 
class IonQConfig:
    """IonQ API configuration"""
    api_key: str = ""
    api_url: str = "https://api.ionq.co"
    api_version: str = "v0.4"
    default_backend: IonQBackend = IonQBackend.SIMULATOR
    default_shots: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0
    poll_interval: float = 2.0
    timeout: float = 300.0  # 5 minutes
    
    def __post_init__(self):
        # Try to load from environment
        if not self.api_key:
            self.api_key = os.environ.get("IONQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("IonQ API key not provided. Set IONQ_API_KEY environment variable.")


class IonQClient:
    """Async IonQ API client"""
    
    def __init__(self, config: Optional[IonQConfig] = None):
        self.config = config or IonQConfig()
        self.base_url = f"{self.config.api_url}/{self.config.api_version}"
        self.headers = {
            "Authorization": f"apiKey {self.config.api_key}",
            "Content-Type": "application/json"
        }
        self.session: Optional[aiohttp.ClientSession] = None
        self.job_cache: Dict[str, IonQJob] = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def _request(self, method: str, endpoint: str, 
                      data: Optional[Dict] = None) -> Dict:
        """Make API request with retries"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.request(
                    method, url, 
                    headers=self.headers,
                    json=data if data else None
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                        continue
                    else:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                        
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay)
                
        raise Exception(f"Max retries exceeded for {endpoint}")
    
    async def submit_circuit(self, 
                           genome: QuantumGenome,
                           backend: Optional[IonQBackend] = None,
                           shots: Optional[int] = None,
                           name: Optional[str] = None,
                           error_mitigation: bool = False) -> IonQJob:
        """Submit a quantum circuit for execution"""
        
        backend = backend or self.config.default_backend
        shots = shots or self.config.default_shots
        
        # Convert genome to IonQ format
        circuit_json = genome.to_circuit_json()
        
        # Build job payload
        job_data = {
            "name": name or f"dfal_gen{genome.generation}_{genome.id[:8]}",
            "target": backend.value,
            "shots": shots,
            "input": circuit_json
        }
        
        # Add optional parameters
        if backend != IonQBackend.SIMULATOR:
            # Add noise model for hardware
            job_data["noise"] = {"model": "harmony"}
            
            if error_mitigation:
                job_data["error_mitigation"] = {
                    "debias": True,
                    "symmetrization": True
                }
                
        # Submit job
        response = await self._request("POST", "jobs", job_data)
        
        # Create job object
        job = IonQJob(
            id=response["id"],
            status=JobStatus(response["status"]),
            backend=backend.value,
            shots=shots,
            created_at=datetime.now(),
            circuit=circuit_json,
            metadata={
                "genome_id": genome.id,
                "generation": genome.generation,
                "native_compliant": genome.native_gate_compliant
            }
        )
        
        self.job_cache[job.id] = job
        logger.info(f"Submitted job {job.id} to {backend.value}")
        
        return job
    
    async def submit_batch(self,
                         genomes: List[QuantumGenome],
                         backend: Optional[IonQBackend] = None,
                         shots: Optional[int] = None,
                         name_prefix: Optional[str] = None) -> List[IonQJob]:
        """Submit multiple circuits as a batch"""
        
        backend = backend or self.config.default_backend
        shots = shots or self.config.default_shots
        
        # IonQ supports multi-circuit jobs
        circuits = [g.to_circuit_json() for g in genomes]
        
        job_data = {
            "name": name_prefix or f"dfal_batch_{datetime.now().isoformat()}",
            "target": backend.value,
            "shots": shots,
            "input": circuits  # Multiple circuits
        }
        
        if backend != IonQBackend.SIMULATOR:
            job_data["noise"] = {"model": "harmony"}
            
        response = await self._request("POST", "jobs", job_data)
        
        # Create job objects for tracking
        jobs = []
        for i, genome in enumerate(genomes):
            job = IonQJob(
                id=f"{response['id']}_{i}",  # Sub-job ID
                status=JobStatus(response["status"]),
                backend=backend.value,
                shots=shots,
                created_at=datetime.now(),
                circuit=circuits[i],
                metadata={
                    "genome_id": genome.id,
                    "generation": genome.generation,
                    "batch_id": response["id"],
                    "batch_index": i
                }
            )
            jobs.append(job)
            self.job_cache[job.id] = job
            
        logger.info(f"Submitted batch of {len(genomes)} circuits as job {response['id']}")
        
        return jobs
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get current job status"""
        # Extract batch ID if this is a sub-job
        batch_id = job_id.split("_")[0] if "_" in job_id else job_id
        
        response = await self._request("GET", f"jobs/{batch_id}")
        status = JobStatus(response["status"])
        
        # Update cache
        if job_id in self.job_cache:
            self.job_cache[job_id].status = status
            
        return status
    
    async def wait_for_job(self, job_id: str, 
                          timeout: Optional[float] = None) -> IonQJob:
        """Wait for job completion"""
        timeout = timeout or self.config.timeout
        start_time = time.time()
        
        while True:
            status = await self.get_job_status(job_id)
            
            if status == JobStatus.COMPLETED:
                return await self.get_results(job_id)
            elif status == JobStatus.FAILED:
                job = self.job_cache.get(job_id)
                if job:
                    job.error = "Job failed on IonQ backend"
                raise Exception(f"Job {job_id} failed")
            elif status == JobStatus.CANCELED:
                raise Exception(f"Job {job_id} was canceled")
                
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")
                
            await asyncio.sleep(self.config.poll_interval)
    
    async def get_results(self, job_id: str) -> IonQJob:
        """Get job results"""
        # Extract batch ID and index if this is a sub-job
        if "_" in job_id:
            batch_id, index = job_id.split("_")
            index = int(index)
        else:
            batch_id = job_id
            index = None
            
        # Get probabilities
        prob_response = await self._request(
            "GET", f"jobs/{batch_id}/results/probabilities"
        )
        
        # Get job details for metadata
        job_response = await self._request("GET", f"jobs/{batch_id}")
        
        # Extract results for specific circuit if batch
        if index is not None and isinstance(prob_response, list):
            probabilities = prob_response[index]
        else:
            probabilities = prob_response
            
        # Process results
        results = {
            "probabilities": probabilities,
            "shots": job_response.get("shots", self.config.default_shots),
            "backend": job_response.get("target", "unknown"),
            "execution_time": job_response.get("execution_time"),
            "gate_counts": job_response.get("gate_counts"),
        }
        
        # Update job in cache
        if job_id in self.job_cache:
            job = self.job_cache[job_id]
            job.status = JobStatus.COMPLETED
            job.results = results
            return job
        else:
            # Create new job object
            return IonQJob(
                id=job_id,
                status=JobStatus.COMPLETED,
                backend=results["backend"],
                shots=results["shots"],
                created_at=datetime.now(),
                circuit={},
                results=results
            )
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        batch_id = job_id.split("_")[0] if "_" in job_id else job_id
        
        try:
            await self._request("PUT", f"jobs/{batch_id}/cancel")
            
            if job_id in self.job_cache:
                self.job_cache[job_id].status = JobStatus.CANCELED
                
            logger.info(f"Canceled job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def get_backend_status(self, backend: IonQBackend) -> Dict:
        """Get backend status and properties"""
        response = await self._request("GET", f"backends/{backend.value}")
        
        return {
            "backend": backend.value,
            "status": response.get("status", "unknown"),
            "qubits": response.get("qubits", 0),
            "connectivity": response.get("connectivity", "unknown"),
            "gate_set": response.get("native_gates", []),
            "average_queue_time": response.get("average_queue_time"),
            "characterization": response.get("characterization", {})
        }
    
    async def estimate_cost(self, 
                           genome: QuantumGenome,
                           backend: IonQBackend,
                           shots: int) -> float:
        """Estimate cost for running a circuit"""
        # Simplified cost model - actual costs vary
        costs = {
            IonQBackend.SIMULATOR: 0.0,
            IonQBackend.ARIA_1: 0.01,  # $0.01 per shot
            IonQBackend.ARIA_2: 0.01,
            IonQBackend.FORTE_1: 0.015,
            IonQBackend.QPU: 0.02
        }
        
        base_cost = costs.get(backend, 0.01)
        circuit_complexity = genome.depth() * genome.two_qubit_gate_count()
        
        # Cost formula (simplified)
        total_cost = base_cost * shots * (1 + circuit_complexity / 100)
        
        return total_cost


class IonQSimulator:
    """Local simulator with IonQ noise models"""
    
    def __init__(self):
        self.has_qiskit = False
        try:
            import qiskit
            from qiskit import Aer
            from qiskit.providers.aer.noise import NoiseModel
            self.has_qiskit = True
            self.backend = Aer.get_backend('qasm_simulator')
        except ImportError:
            logger.warning("Qiskit not installed. Using basic simulator.")
            
    def simulate(self, genome: QuantumGenome, 
                shots: int = 1000,
                noise_model: Optional[str] = None) -> Dict:
        """Simulate circuit locally"""
        
        if self.has_qiskit:
            return self._qiskit_simulate(genome, shots, noise_model)
        else:
            return self._basic_simulate(genome, shots)
            
    def _qiskit_simulate(self, genome: QuantumGenome, 
                        shots: int,
                        noise_model: Optional[str]) -> Dict:
        """Simulate using Qiskit"""
        from qiskit import execute, QuantumCircuit, ClassicalRegister
        
        # Convert to Qiskit circuit
        qc = genome.to_qiskit_circuit()
        
        # Add measurements
        cr = ClassicalRegister(genome.qubit_count)
        qc.add_register(cr)
        qc.measure_all()
        
        # Execute
        job = execute(qc, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Convert to probabilities
        probabilities = {
            state: count / shots 
            for state, count in counts.items()
        }
        
        # Get state vector if possible
        state_vector = None
        if shots == 1:
            sv_backend = Aer.get_backend('statevector_simulator')
            qc_no_measure = genome.to_qiskit_circuit()
            job = execute(qc_no_measure, sv_backend)
            state_vector = job.result().get_statevector().data
            
        return {
            "probabilities": probabilities,
            "counts": counts,
            "state_vector": state_vector,
            "shots": shots
        }
    
    def _basic_simulate(self, genome: QuantumGenome, shots: int) -> Dict:
        """Basic simulation without Qiskit"""
        # Very simplified - just return random distribution
        n_states = 2 ** genome.qubit_count
        
        # Generate random probabilities
        probs = np.random.dirichlet(np.ones(n_states))
        
        # Convert to bit string format
        probabilities = {}
        for i in range(n_states):
            bitstring = format(i, f'0{genome.qubit_count}b')
            probabilities[bitstring] = probs[i]
            
        return {
            "probabilities": probabilities,
            "shots": shots,
            "warning": "Using basic simulator - results are random"
        }


class IonQBatchProcessor:
    """Process multiple circuits efficiently"""
    
    def __init__(self, client: IonQClient, max_concurrent: int = 10):
        self.client = client
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
    async def process_generation(self,
                                genomes: List[QuantumGenome],
                                backend: IonQBackend = IonQBackend.SIMULATOR,
                                shots: int = 1000) -> List[Dict]:
        """Process an entire generation of circuits"""
        
        results = []
        
        # Split into batches for IonQ
        batch_size = 10  # IonQ limit
        batches = [genomes[i:i+batch_size] 
                  for i in range(0, len(genomes), batch_size)]
        
        # Submit all batches
        all_jobs = []
        for batch in batches:
            jobs = await self.client.submit_batch(batch, backend, shots)
            all_jobs.extend(jobs)
            
        # Wait for all jobs
        tasks = [self.client.wait_for_job(job.id) for job in all_jobs]
        completed_jobs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract results
        for job in completed_jobs:
            if isinstance(job, Exception):
                results.append({"error": str(job)})
            else:
                results.append(job.results)
                
        return results
    
    async def adaptive_backend_selection(self,
                                        genome: QuantumGenome) -> IonQBackend:
        """Select best backend based on circuit properties"""
        
        # Simple heuristic
        if genome.qubit_count <= 3 and genome.depth() < 10:
            # Small circuit - use simulator
            return IonQBackend.SIMULATOR
        elif genome.native_gate_compliant and genome.qubit_count <= 11:
            # Native gates and fits on Aria
            return IonQBackend.ARIA_1
        elif genome.qubit_count <= 32:
            # Larger circuit - needs Forte
            return IonQBackend.FORTE_1
        else:
            # Very large - simulator only
            return IonQBackend.SIMULATOR