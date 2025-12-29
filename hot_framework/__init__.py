"""
HOT Framework - Hardware-Optimized Techniques for Quantum Circuit Execution

A systematic methodology for quantum circuit optimization that operates above standard SDK compilation.
HOT addresses three interconnected concerns:
- H: Hardware-aware error mitigation through topological scaffolding
- O: Optimal island selection via calibration-driven qubit clustering  
- T: Tidy measurement hygiene to minimize crosstalk and extraneous readout
"""

from .compiler import HOTCompiler
from .core import (
    HOTResult,
    Island,
    Scaffold,
    OrphanPolicy,
    MeasurementPolicy,
    JobVariants,
    AlgorithmSpec,
    AlgorithmClass,
    CalibrationData,
    DistributionMetrics,
)
from .mitigation import (
    MitigationHook,
    DynamicalDecoupling,
    ZeroNoiseExtrapolation,
)
from .variational import (
    VariationalCoDesigner,
    RetuningConfig,
    CoDesignResult,
)
from .measurement import (
    MeasurementClass,
    classify_measurements,
    apply_measurement_policy,
)
from .utils import (
    compute_distribution_metrics,
    run_controlled_comparison,
)

__version__ = "2.0.0"
__all__ = [
    "HOTCompiler",
    "HOTResult", 
    "Island",
    "Scaffold",
    "OrphanPolicy",
    "MeasurementPolicy",
    "JobVariants",
    "AlgorithmSpec",
    "AlgorithmClass",
    "CalibrationData",
    "DistributionMetrics",
    "MitigationHook",
    "DynamicalDecoupling", 
    "ZeroNoiseExtrapolation",
    "VariationalCoDesigner",
    "RetuningConfig",
    "CoDesignResult",
    "MeasurementClass",
    "classify_measurements",
    "apply_measurement_policy",
    "compute_distribution_metrics",
    "run_controlled_comparison",
]
