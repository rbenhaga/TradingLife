"""Core modules for crypto trading bot"""

from src.core.trading_bot import TradingBot
from src.core.weighted_score_engine import WeightedScoreEngine
from src.core.multi_pair_manager import MultiPairManager
from src.core.watchlist_scanner import WatchlistScanner
from src.core.risk_manager import RiskManager
from src.core.market_data import MarketData
from src.core.websocket_market_feed import WebSocketMarketFeed, MultiExchangeFeed, DataType, MarketUpdate, OrderBookSnapshot

__all__ = [
    'TradingBot',
    'WeightedScoreEngine',
    'MultiPairManager',
    'WatchlistScanner',
    'RiskManager',
    'MarketData',
    'WebSocketMarketFeed',
    'MultiExchangeFeed',
    'DataType',
    'MarketUpdate',
    'OrderBookSnapshot'
]