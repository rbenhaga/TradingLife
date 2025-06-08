"""Trading strategies"""

from src.strategy import Strategy, MultiSignalStrategy
from src.multi_signal import MultiSignalStrategy as MultiSignal

__all__ = ['Strategy', 'MultiSignalStrategy', 'MultiSignal']