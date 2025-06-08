"""Trading strategies"""

# Import depuis strategy.py qui contient toutes les classes
from .strategy import Strategy, MultiSignalStrategy

# Import depuis multi_signal.py si il existe (pour compatibilité)
try:
    from .multi_signal import MultiSignalStrategy as MultiSignal
except ImportError:
    # Si multi_signal.py n'existe pas, utiliser MultiSignalStrategy
    MultiSignal = MultiSignalStrategy

__all__ = ['Strategy', 'MultiSignalStrategy', 'MultiSignal']