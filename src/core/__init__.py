# src/core/__init__.py
"""Modules principaux du bot de trading"""

from .logger import log_info, log_error, log_debug, log_warning, log_trade, log_performance
from .watchlist import DynamicWatchlist
from .multi_pair_manager import MultiPairManager

__all__ = [
    'log_info', 
    'log_error', 
    'log_debug', 
    'log_warning',
    'log_trade',
    'log_performance',
    'DynamicWatchlist',
    'MultiPairManager'
]