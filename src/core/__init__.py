"""Core modules for crypto trading bot"""

from .trading_bot import TradingBot
from .weighted_score_engine import WeightedScoreEngine
from .multi_pair_manager import MultiPairManager
from .watchlist_scanner import WatchlistScanner
from .backtester import Backtester
from .weight_optimizer import WeightOptimizer

__all__ = [
    'TradingBot',
    'WeightedScoreEngine',
    'MultiPairManager',
    'WatchlistScanner',
    'Backtester',
    'WeightOptimizer',
    'MarketData',
    'RiskManager',
]
