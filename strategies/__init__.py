"""
Strategy modules for different query types
"""

from .base_strategy import BaseStrategy
from .fact_strategy import SimpleFactStrategy, ComplexFactStrategy
from .summary_strategy import SummaryStrategy
from .comparison_strategy import ComparisonStrategy
from .reasoning_strategy import ReasoningStrategy

__all__ = [
    "BaseStrategy",
    "SimpleFactStrategy",
    "ComplexFactStrategy",
    "SummaryStrategy",
    "ComparisonStrategy",
    "ReasoningStrategy"
]
