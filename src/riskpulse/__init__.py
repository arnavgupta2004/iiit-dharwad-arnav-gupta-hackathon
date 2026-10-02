"""RiskPulse: AI/NLP risk engine with a sentiment index rebalancer and a stress tester."""

from riskpulse import _storage

_storage.configure()  # external caches (.env), before any Hugging Face import

__version__ = "0.1.0"
