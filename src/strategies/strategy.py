"""
Amélioration de la classe Strategy avec les méthodes manquantes
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
from datetime import datetime
from ..core.logger import log_info, log_debug, log_warning, log_error
from ..core.weighted_score_engine import WeightedScoreEngine

class Strategy(ABC):
    """Classe de base améliorée pour les stratégies de trading"""
    
    def __init__(self, symbol: str, timeframe: str = '15m'):
        """
        Initialise la stratégie
        
        Args:
            symbol: Symbole de trading (ex: 'BTC/USDT')
            timeframe: Période des bougies
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logging.getLogger(f'Strategy.{self.__class__.__name__}')
        
        # État de la stratégie
        self.current_position = None  # 'LONG', 'SHORT', ou None
        self.entry_price = None
        self.entry_time = None
        self.stop_loss = None
        self.take_profit = None
        
        self.logger.info(f"Stratégie {self.__class__.__name__} initialisée pour {symbol}")
    
    @abstractmethod
    def should_enter(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Détermine si on doit entrer en position
        
        Args:
            df: DataFrame avec les données de marché et indicateurs
            
        Returns:
            Dict avec signal d'entrée ou None
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
        pass
    
    @abstractmethod
    def should_exit(self, df: pd.DataFrame, position: Dict) -> Optional[Dict]:
        """
        Détermine si on doit sortir de position
        
        Args:
            df: DataFrame avec les données de marché
            position: Dict avec les infos de la position actuelle
                {
                    'entry_price': float,
                    'current_price': float,
                    'quantity': float,
                    'side': 'long' ou 'short',
                    'entry_time': datetime,
                    'unrealized_pnl': float,
                    'unrealized_pnl_pct': float
                }
            
        Returns:
            Dict avec signal de sortie ou None
            Format: {
                'action': 'SELL' ou 'BUY' (opposé de la position),
                'type': 'market' ou 'limit',
                'price': float (si limit),
                'reason': str ('take_profit', 'stop_loss', 'signal', etc.)
            }
        """
        pass
      
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyse complète retournant l'état de la stratégie
        
        Args:
            df: DataFrame avec les données
            
        Returns:
            Dict avec l'analyse complète
        """
        if len(df) < 20:
            return {
                'action': 'NEUTRAL',
                'confidence': 0,
                'reason': 'Insufficient data',
                'indicators': {}
            }
        
        # Si pas de position, chercher un signal d'entrée
        if self.current_position is None:
            entry_signal = self.should_enter(df)
            if entry_signal and isinstance(entry_signal, dict):
                return {
                    'action': entry_signal.get('action', 'NEUTRAL'),
                    'confidence': entry_signal.get('confidence', 0.5),
                    'reason': entry_signal.get('reason', 'Signal detected'),
                    'stop_loss': entry_signal.get('stop_loss'),
                    'take_profit': entry_signal.get('take_profit'),
                    'type': entry_signal.get('type', 'market')
                }
        
        # Si position ouverte, chercher un signal de sortie
        else:
            current_price = df['close'].iloc[-1]
            position_info = {
                'entry_price': self.entry_price,
                'current_price': current_price,
                'side': self.current_position.lower(),
                'entry_time': self.entry_time,
                'unrealized_pnl_pct': ((current_price / self.entry_price) - 1) * 100
            }
            
            exit_signal = self.should_exit(df, position_info)
            if exit_signal and isinstance(exit_signal, dict):
                return {
                    'action': exit_signal.get('action', 'NEUTRAL'),
                    'confidence': 1.0,  # Sortie toujours haute confiance
                    'reason': exit_signal.get('reason', 'Exit signal'),
                    'type': exit_signal.get('type', 'market')
                }
        
        # Pas de signal
        return {
            'action': 'NEUTRAL',
            'confidence': 0,
            'reason': 'No clear signal',
            'position': self.current_position
        }

    def get_position_size(self, capital: float, current_price: float, 
                        risk_per_trade: float = 0.02) -> float:
        """
        Calcule la taille de position basée sur le risque
        
        Args:
            capital: Capital disponible
            current_price: Prix actuel
            risk_per_trade: Risque par trade (défaut 2%)
            
        Returns:
            Taille de position en unités
        """
        # Position sizing basé sur le risque
        risk_amount = capital * risk_per_trade
        
        # Si on a un stop loss défini, l'utiliser
        if self.stop_loss and self.entry_price:
            stop_distance = abs(self.entry_price - self.stop_loss) / self.entry_price
            position_value = risk_amount / stop_distance
        else:
            # Sinon, utiliser un stop par défaut de 5%
            position_value = risk_amount / 0.05
        
        # Limiter à 10% du capital max
        position_value = min(position_value, capital * 0.1)
        
        # Convertir en unités
        position_size = position_value / current_price
        
        return position_size


class MultiSignalStrategy(Strategy):
    """Stratégie multi-signaux améliorée"""
    
    def __init__(self, symbol: str, timeframe: str = '15m',
                 score_weights: Dict[str, float] = None):
        """
        Initialise la stratégie multi-signal
        
        Args:
            symbol: Symbole de trading
            timeframe: Période des bougies
            score_weights: Poids personnalisés pour chaque indicateur
        """
        super().__init__(symbol, timeframe)
        
        # Import du score engine
        self.score_engine = WeightedScoreEngine(score_weights)
        
        # Paramètres de la stratégie
        self.entry_threshold = 0.5  # Score minimum pour entrer
        self.exit_threshold = -0.3  # Score pour sortir
        self.stop_loss_pct = 0.05   # 5% stop loss
        self.take_profit_pct = 0.10 # 10% take profit
        
        # Paramètres RSI
        self.rsi_oversold = 30
        self.rsi_overbought = 70
    
    def should_enter(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Détermine si on doit entrer en position
        
        Args:
            df: DataFrame avec les données et indicateurs
            
        Returns:
            Signal d'entrée ou None
        """
        if self.current_position is not None:
            return None
        
        # Analyser avec le score engine
        signals = self.score_engine.analyze_indicators(df)
        score_result = self.score_engine.calculate_score(signals)
        
        score = score_result['score']
        confidence = score_result['confidence']
        
        # Signal d'achat fort
        if score > self.entry_threshold and confidence > 0.6:
            current_price = df['close'].iloc[-1]
            
            return {
                'action': 'BUY',
                'type': 'market',
                'confidence': confidence,
                'reason': f"Score élevé: {score:.2f}",
                'stop_loss': current_price * (1 - self.stop_loss_pct),
                'take_profit': current_price * (1 + self.take_profit_pct),
                'score': score,
                'details': score_result['details']
            }
        
        # Signal de vente fort (short)
        elif score < -self.entry_threshold and confidence > 0.6:
            current_price = df['close'].iloc[-1]
            
            return {
                'action': 'SELL',
                'type': 'market',
                'confidence': confidence,
                'reason': f"Score faible: {score:.2f}",
                'stop_loss': current_price * (1 + self.stop_loss_pct),
                'take_profit': current_price * (1 - self.take_profit_pct),
                'score': score,
                'details': score_result['details']
            }
        
        return None
    
    def should_exit(self, df: pd.DataFrame, position: Dict) -> Optional[Dict]:
        """
        Détermine si on doit sortir de position
        
        Args:
            df: DataFrame avec les données
            position: Informations sur la position actuelle
            
        Returns:
            Signal de sortie ou None
        """
        current_price = position['current_price']
        entry_price = position['entry_price']
        side = position['side']
        pnl_pct = position['unrealized_pnl_pct']
        
        # Vérifier stop loss
        if side == 'long' and current_price <= self.stop_loss:
            return {
                'action': 'SELL',
                'type': 'market',
                'reason': 'Stop loss atteint'
            }
        elif side == 'short' and current_price >= self.stop_loss:
            return {
                'action': 'BUY',
                'type': 'market',
                'reason': 'Stop loss atteint'
            }
        
        # Vérifier take profit
        if side == 'long' and current_price >= self.take_profit:
            return {
                'action': 'SELL',
                'type': 'market',
                'reason': 'Take profit atteint'
            }
        elif side == 'short' and current_price <= self.take_profit:
            return {
                'action': 'BUY',
                'type': 'market',
                'reason': 'Take profit atteint'
            }
        
        # Analyser le score actuel
        signals = self.score_engine.analyze_indicators(df)
        score_result = self.score_engine.calculate_score(signals)
        score = score_result['score']
        
        # Sortie sur inversion de signal
        if side == 'long' and score < self.exit_threshold:
            return {
                'action': 'SELL',
                'type': 'market',
                'reason': f'Signal inversé (score: {score:.2f})'
            }
        elif side == 'short' and score > -self.exit_threshold:
            return {
                'action': 'BUY',
                'type': 'market',
                'reason': f'Signal inversé (score: {score:.2f})'
            }
        
        # Trailing stop si en profit
        if pnl_pct > 5:  # Si profit > 5%
            # Ajuster le stop loss pour protéger les profits
            if side == 'long':
                new_stop = current_price * 0.97  # 3% en dessous
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
            else:  # short
                new_stop = current_price * 1.03  # 3% au dessus
                if new_stop < self.stop_loss:
                    self.stop_loss = new_stop
        
        return None
    
    def get_indicators(self) -> Dict:
        """
        Retourne l'état actuel des indicateurs
        
        Returns:
            Dict avec les indicateurs et l'état
        """
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'position': self.current_position,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strategy': 'MultiSignal'
        }