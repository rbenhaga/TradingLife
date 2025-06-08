"""
Classe de base pour les stratégies de trading
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from src.core.logger import log_info, log_debug

class Strategy(ABC):
    """Classe abstraite pour les stratégies de trading"""
    
    def __init__(self, symbol: str, timeframe: str = '15m'):
        """
        Initialise la stratégie
        
        Args:
            symbol: Symbole de trading (ex: 'BTC/USDT')
            timeframe: Période des bougies (ex: '15m', '1h', '4h')
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = pd.DataFrame()
        self.last_signal = None
        self.last_signal_time = None
        
        log_info(f"Stratégie initialisée pour {symbol} ({timeframe})")
    
    @abstractmethod
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les signaux de trading
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec les signaux ajoutés
        """
        pass
    
    @abstractmethod
    def should_buy(self, df: pd.DataFrame) -> bool:
        """
        Détermine si on devrait acheter
        
        Args:
            df: DataFrame avec les données et signaux
            
        Returns:
            True si on devrait acheter
        """
        pass
    
    @abstractmethod
    def should_sell(self, df: pd.DataFrame) -> bool:
        """
        Détermine si on devrait vendre
        
        Args:
            df: DataFrame avec les données et signaux
            
        Returns:
            True si on devrait vendre
        """
        pass
    
    def update(self, df: pd.DataFrame):
        """
        Met à jour les données et calcule les signaux
        
        Args:
            df: DataFrame avec les données OHLCV
        """
        if len(df) < 2:
            log_debug(f"{self.symbol} - Pas assez de données")
            return
        
        # Calculer les signaux
        self.data = self.calculate_signals(df)
        
        # Vérifier les signaux
        current_time = df.index[-1]
        
        if self.should_buy(df):
            if self.last_signal != 'BUY':
                self.last_signal = 'BUY'
                self.last_signal_time = current_time
                log_info(f"📈 Signal d'achat pour {self.symbol}")
        elif self.should_sell(df):
            if self.last_signal != 'SELL':
                self.last_signal = 'SELL'
                self.last_signal_time = current_time
                log_info(f"📉 Signal de vente pour {self.symbol}")
        else:
            self.last_signal = None
    
    def get_current_signal(self) -> Optional[str]:
        """
        Retourne le signal actuel
        
        Returns:
            'BUY', 'SELL' ou None
        """
        return self.last_signal
    
    def get_signal_time(self) -> Optional[datetime]:
        """
        Retourne le timestamp du dernier signal
        
        Returns:
            Timestamp du dernier signal ou None
        """
        return self.last_signal_time
    
    def get_indicators(self) -> Dict:
        """
        Retourne les indicateurs actuels
        
        Returns:
            Dictionnaire des indicateurs
        """
        if len(self.data) == 0:
            return {}
        
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'current_price': float(self.data['close'].iloc[-1]),
            'last_signal': self.last_signal,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None
        } 