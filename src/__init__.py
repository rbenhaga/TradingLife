"""
TradingLife - Bot de trading crypto haute performance
"""

__version__ = "1.0.0"
__author__ = "TheRedFiire"

from src.core.trading_bot import TradingBot, BotState
from src.core.enhanced_trading_bot import EnhancedTradingBot

__all__ = ['TradingBot', 'BotState', 'EnhancedTradingBot']
