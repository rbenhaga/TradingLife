"""Core modules for the trading bot"""

from .trading_bot import TradingBot, BotState
from .weighted_score_engine import WeightedScoreEngine
from .multi_pair_manager import MultiPairManager
from .watchlist_scanner import WatchlistScanner
from .risk_manager import RiskManager
from .market_data import MarketData
from .websocket_market_feed import WebSocketMarketFeed, DataType, MarketUpdate, OrderBookSnapshot
from .adaptive_backtester import AdaptiveBacktester
from .logger import log_info, log_error, log_warning, log_debug

__all__ = [
    'TradingBot',
    'BotState',
    'WeightedScoreEngine',
    'MultiPairManager',
    'WatchlistScanner',
    'RiskManager',
    'MarketData',
    'WebSocketMarketFeed',
    'DataType',
    'MarketUpdate',
    'OrderBookSnapshot',
    'AdaptiveBacktester',
    'log_info',
    'log_error',
    'log_warning',
    'log_debug'
]