"""
Stratégie de trading basée sur plusieurs signaux techniques
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from .strategy import Strategy
from ..core.weighted_score_engine import WeightedScoreEngine
from src.core.logger import log_info, log_debug

class MultiSignalStrategy(Strategy):
    """Stratégie combinant plusieurs indicateurs techniques"""
    
    def __init__(self, symbol: str, timeframe: str = '15m',
                 score_weights: Dict[str, float] = None):
        """
        Initialise la stratégie multi-signal
        
        Args:
            symbol: Symbole de trading (ex: 'BTC/USDT')
            timeframe: Période des bougies
            score_weights: Poids personnalisés pour chaque indicateur
        """
        super().__init__(symbol, timeframe)
        self.score_engine = WeightedScoreEngine(score_weights)
        
        # Définir les seuils RSI
        self.rsi_oversold = 30  # Zone de survente
        self.rsi_overbought = 70  # Zone de surachat
        
        self.logger.info(f"Stratégie Multi-Signal avec score pondéré initialisée pour {symbol}")
    
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les signaux de trading
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec les signaux ajoutés
        """
        # Calculer les indicateurs
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
        
        # Calculer le score pondéré
        df['score'] = self.score_engine.calculate_score(df)
        
        # Générer les signaux
        df['signal'] = 0
        df.loc[df['score'] > 0.7, 'signal'] = 1  # Signal d'achat
        df.loc[df['score'] < -0.7, 'signal'] = -1  # Signal de vente
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series, 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> tuple:
        """Calcule le MACD"""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal
    
    def calculate_bollinger_bands(self, prices: pd.Series, 
                                period: int = 20, 
                                std_dev: float = 2.0) -> tuple:
        """Calcule les bandes de Bollinger"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def should_buy(self, df: pd.DataFrame) -> bool:
        """
        Détermine si on devrait acheter
        
        Args:
            df: DataFrame avec les données et signaux
            
        Returns:
            True si on devrait acheter
        """
        if len(df) < 2:
            return False
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Vérifier les conditions d'achat
        rsi_condition = current['rsi'] < self.rsi_oversold
        macd_condition = current['macd'] > current['macd_signal']
        bb_condition = current['close'] < current['bb_lower']
        score_condition = current['score'] > 0.7
        
        return rsi_condition and (macd_condition or bb_condition) and score_condition
    
    def should_sell(self, df: pd.DataFrame) -> bool:
        """
        Détermine si on devrait vendre
        
        Args:
            df: DataFrame avec les données et signaux
            
        Returns:
            True si on devrait vendre
        """
        if len(df) < 2:
            return False
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Vérifier les conditions de vente
        rsi_condition = current['rsi'] > self.rsi_overbought
        macd_condition = current['macd'] < current['macd_signal']
        bb_condition = current['close'] > current['bb_upper']
        score_condition = current['score'] < -0.7
        
        return rsi_condition and (macd_condition or bb_condition) and score_condition