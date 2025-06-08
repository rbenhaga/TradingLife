"""Core modules for crypto trading bot"""

from src.trading_bot import TradingBot
from src.weighted_score_engine import WeightedScoreEngine
from src.multi_pair_manager import MultiPairManager
from src.watchlist_scanner import WatchlistScanner
from src.backtester import Backtester
from src.weight_optimizer import WeightOptimizer
from src.market_data import MarketData
from src.risk_manager import RiskManager

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
