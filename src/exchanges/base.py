"""
Module de base pour les connecteurs d'exchange
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

class BaseExchange(ABC):
    """Classe de base pour les connecteurs d'exchange"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Établit la connexion avec l'exchange"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict:
        """Récupère le ticker pour un symbole"""
        pass
    
    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Récupère les données OHLCV"""
        pass 