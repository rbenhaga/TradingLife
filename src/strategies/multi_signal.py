"""
Stratégie de trading basée sur plusieurs signaux techniques
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from .strategy import Strategy
from ..core.weighted_score_engine import WeightedScoreEngine
from ..core.logger import log_info, log_debug, log_warning
from ..utils.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands
from ..utils.helpers import calculate_position_size

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

        self.indicators = {
            'rsi': {'weight': 0.2, 'period': 14},
            'macd': {'weight': 0.3, 'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'weight': 0.3, 'period': 20, 'std': 2.0},
            'volume': {'weight': 0.2, 'period': 20}
        }
    
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
    
    def should_enter(self, data: dict) -> Optional[Dict]:
        """
        Détermine si on doit entrer en position
        
        Args:
            data (dict): Données de marché et indicateurs
            
        Returns:
            Optional[Dict]: Dictionnaire avec le signal d'entrée ou None
            Format: {
                'action': 'BUY' ou 'SELL',
                'type': 'market' ou 'limit',
                'price': float (si limit),
                'confidence': float (0-1),
                'reason': str,
                'stop_loss': float,
                'take_profit': float
            }
        """
        # Calculer le score pondéré
        score = self.score_engine.calculate_score(data, self.indicators)
        
        # Conditions d'entrée
        if score > 0.7:  # Signal d'achat fort
            return {
                'action': 'BUY',
                'type': 'market',
                'confidence': score,
                'reason': "Signal d'achat fort",
                'stop_loss': data['close'] * 0.95,  # 5% stop loss
                'take_profit': data['close'] * 1.15  # 15% take profit
            }
        elif score > 0.5:  # Signal d'achat modéré
            return {
                'action': 'BUY',
                'type': 'market',
                'confidence': score,
                'reason': "Signal d'achat modéré",
                'stop_loss': data['close'] * 0.97,  # 3% stop loss
                'take_profit': data['close'] * 1.10  # 10% take profit
            }
            
        return None

    def should_exit(self, data: dict, position: dict) -> Optional[Dict]:
        """
        Détermine si on doit sortir de position
        
        Args:
            data (dict): Données de marché et indicateurs
            position (dict): Informations sur la position actuelle
            
        Returns:
            Optional[Dict]: Dictionnaire avec le signal de sortie ou None
            Format: {
                'action': 'SELL' ou 'BUY' (opposé de la position),
                'type': 'market' ou 'limit',
                'price': float (si limit),
                'reason': str ('take_profit', 'stop_loss', 'signal', etc.)
            }
        """
        # Calculer le score pondéré
        score = self.score_engine.calculate_score(data, self.indicators)
        
        # Conditions de sortie
        if score < -0.7:  # Signal de vente fort
            return {
                'action': 'SELL' if position['side'] == 'long' else 'BUY',
                'type': 'market',
                'reason': "Signal de vente fort"
            }
        elif score < -0.5:  # Signal de vente modéré
            return {
                'action': 'SELL' if position['side'] == 'long' else 'BUY',
                'type': 'market',
                'reason': "Signal de vente modéré"
            }
            
        return None