"""
Module de backtesting pour les stratégies de trading
"""

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from src.logger import log_info, log_error, log_debug
from src.strategies.strategy import Strategy

class Backtester:
    """Classe pour le backtesting des stratégies"""
    
    def __init__(self):
        """Initialise le backtester"""
        pass 