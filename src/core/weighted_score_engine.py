"""
Moteur de score pondéré multi-indicateurs
Combine plusieurs signaux techniques avec des poids optimisables
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from src.core.logger import log_info, log_debug, log_warning

@dataclass
class Signal:
    """Représente un signal individuel"""
    name: str
    value: float  # Entre -1 et 1
    weight: float
    confidence: float  # Entre 0 et 1
    reason: str
    
    @property
    def weighted_value(self) -> float:
        """Valeur pondérée du signal"""
        return self.value * self.weight * self.confidence

@dataclass
class TradingScore:
    """Score de trading composite"""
    total_score: float
    direction: str  # 'BUY', 'SELL', 'NEUTRAL'
    confidence: float
    signals: List[Signal]
    timestamp: datetime
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour logging"""
        return {
            'total_score': self.total_score,
            'direction': self.direction,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'signals': [
                {
                    'name': s.name,
                    'value': s.value,
                    'weight': s.weight,
                    'confidence': s.confidence,
                    'weighted_value': s.weighted_value,
                    'reason': s.reason
                }
                for s in self.signals
            ]
        }

class WeightedScoreEngine:
    """Moteur de calcul de score pondéré pour les décisions de trading"""
    
    def __init__(self, symbol: str = 'BTC/USDT'):
        self.symbol = symbol
        
        # Poids par défaut (somme = 1.0)
        self.weights = {
            'rsi': 0.20,
            'bollinger': 0.20,
            'macd': 0.15,
            'volume': 0.15,
            'ma_cross': 0.10,
            'momentum': 0.10,
            'volatility': 0.10
        }
        
        # Seuils de décision
        self.buy_threshold = 0.3
        self.strong_buy_threshold = 0.5
        self.sell_threshold = -0.3
        self.strong_sell_threshold = -0.5
        
        # Cache des indicateurs
        self.indicators_cache = {}
        
        log_info(f"Score Engine initialisé pour {symbol} avec poids: {self.weights}")
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calcule tous les indicateurs techniques nécessaires"""
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Moving Averages
        df['ma_fast'] = df['close'].rolling(window=10).mean()
        df['ma_slow'] = df['close'].rolling(window=30).mean()
        df['ma_diff'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow'] * 100
        
        # Momentum
        df['momentum'] = df['close'].pct_change(periods=10) * 100
        
        # Volatility (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _evaluate_rsi(self, rsi: float) -> Signal:
        """Évalue le signal RSI"""
        if pd.isna(rsi):
            return Signal('rsi', 0, self.weights['rsi'], 0, 'RSI non disponible')
        
        # Zones extrêmes
        if rsi < 20:
            return Signal('rsi', 1.0, self.weights['rsi'], 0.9, f'RSI très survendu ({rsi:.1f})')
        elif rsi < 30:
            return Signal('rsi', 0.7, self.weights['rsi'], 0.8, f'RSI survendu ({rsi:.1f})')
        elif rsi < 40:
            return Signal('rsi', 0.3, self.weights['rsi'], 0.6, f'RSI légèrement survendu ({rsi:.1f})')
        elif rsi > 80:
            return Signal('rsi', -1.0, self.weights['rsi'], 0.9, f'RSI très suracheté ({rsi:.1f})')
        elif rsi > 70:
            return Signal('rsi', -0.7, self.weights['rsi'], 0.8, f'RSI suracheté ({rsi:.1f})')
        elif rsi > 60:
            return Signal('rsi', -0.3, self.weights['rsi'], 0.6, f'RSI légèrement suracheté ({rsi:.1f})')
        else:
            return Signal('rsi', 0, self.weights['rsi'], 0.3, f'RSI neutre ({rsi:.1f})')
    
    def _evaluate_bollinger(self, bb_position: float, close: float, bb_upper: float, bb_lower: float) -> Signal:
        """Évalue le signal Bollinger Bands"""
        if pd.isna(bb_position):
            return Signal('bollinger', 0, self.weights['bollinger'], 0, 'Bollinger non disponible')
        
        # Position relative dans les bandes
        if bb_position < 0:  # En dessous de la bande inférieure
            return Signal('bollinger', 1.0, self.weights['bollinger'], 0.95, 'Prix sous bande inférieure')
        elif bb_position < 0.2:
            return Signal('bollinger', 0.6, self.weights['bollinger'], 0.8, 'Prix près bande inférieure')
        elif bb_position > 1:  # Au-dessus de la bande supérieure
            return Signal('bollinger', -1.0, self.weights['bollinger'], 0.95, 'Prix sur bande supérieure')
        elif bb_position > 0.8:
            return Signal('bollinger', -0.6, self.weights['bollinger'], 0.8, 'Prix près bande supérieure')
        else:
            confidence = 1 - abs(bb_position - 0.5) * 2  # Plus confiant aux extrêmes
            return Signal('bollinger', 0, self.weights['bollinger'], confidence * 0.5, f'Prix centré ({bb_position:.2f})')
    
    def _evaluate_macd(self, macd: float, signal: float, histogram: float) -> Signal:
        """Évalue le signal MACD"""
        if pd.isna(macd) or pd.isna(signal):
            return Signal('macd', 0, self.weights['macd'], 0, 'MACD non disponible')
        
        # Croisements et divergences
        if histogram > 0:
            if histogram > abs(macd) * 0.1:  # Histogram significatif
                return Signal('macd', 0.8, self.weights['macd'], 0.85, 'MACD croisement haussier fort')
            else:
                return Signal('macd', 0.4, self.weights['macd'], 0.6, 'MACD haussier')
        else:
            if abs(histogram) > abs(macd) * 0.1:
                return Signal('macd', -0.8, self.weights['macd'], 0.85, 'MACD croisement baissier fort')
            else:
                return Signal('macd', -0.4, self.weights['macd'], 0.6, 'MACD baissier')
    
    def _evaluate_volume(self, volume_ratio: float, price_change: float) -> Signal:
        """Évalue le signal de volume"""
        if pd.isna(volume_ratio):
            return Signal('volume', 0, self.weights['volume'], 0, 'Volume non disponible')
        
        # Volume anormal avec direction du prix
        if volume_ratio > 2.0:
            if price_change > 0:
                return Signal('volume', 0.8, self.weights['volume'], 0.9, f'Volume élevé haussier ({volume_ratio:.1f}x)')
            else:
                return Signal('volume', -0.8, self.weights['volume'], 0.9, f'Volume élevé baissier ({volume_ratio:.1f}x)')
        elif volume_ratio > 1.5:
            if price_change > 0:
                return Signal('volume', 0.4, self.weights['volume'], 0.7, f'Volume au-dessus moyenne ({volume_ratio:.1f}x)')
            else:
                return Signal('volume', -0.4, self.weights['volume'], 0.7, f'Volume au-dessus moyenne ({volume_ratio:.1f}x)')
        elif volume_ratio < 0.5:
            return Signal('volume', 0, self.weights['volume'], 0.3, f'Volume faible ({volume_ratio:.1f}x)')
        else:
            return Signal('volume', 0, self.weights['volume'], 0.5, f'Volume normal ({volume_ratio:.1f}x)')
    
    def _evaluate_ma_cross(self, ma_diff: float, ma_fast: float, ma_slow: float) -> Signal:
        """Évalue le signal de croisement de moyennes mobiles"""
        if pd.isna(ma_diff):
            return Signal('ma_cross', 0, self.weights['ma_cross'], 0, 'MA non disponible')
        
        # Force du croisement
        if ma_diff > 2:
            return Signal('ma_cross', 0.9, self.weights['ma_cross'], 0.85, f'MA rapide >> lente ({ma_diff:.1f}%)')
        elif ma_diff > 0.5:
            return Signal('ma_cross', 0.5, self.weights['ma_cross'], 0.7, f'MA rapide > lente ({ma_diff:.1f}%)')
        elif ma_diff < -2:
            return Signal('ma_cross', -0.9, self.weights['ma_cross'], 0.85, f'MA rapide << lente ({ma_diff:.1f}%)')
        elif ma_diff < -0.5:
            return Signal('ma_cross', -0.5, self.weights['ma_cross'], 0.7, f'MA rapide < lente ({ma_diff:.1f}%)')
        else:
            return Signal('ma_cross', 0, self.weights['ma_cross'], 0.4, f'MA proches ({ma_diff:.1f}%)')
    
    def _evaluate_momentum(self, momentum: float) -> Signal:
        """Évalue le signal de momentum"""
        if pd.isna(momentum):
            return Signal('momentum', 0, self.weights['momentum'], 0, 'Momentum non disponible')
        
        # Force du momentum
        if momentum > 5:
            return Signal('momentum', 0.9, self.weights['momentum'], 0.9, f'Momentum très fort ({momentum:.1f}%)')
        elif momentum > 2:
            return Signal('momentum', 0.5, self.weights['momentum'], 0.7, f'Momentum positif ({momentum:.1f}%)')
        elif momentum < -5:
            return Signal('momentum', -0.9, self.weights['momentum'], 0.9, f'Momentum très négatif ({momentum:.1f}%)')
        elif momentum < -2:
            return Signal('momentum', -0.5, self.weights['momentum'], 0.7, f'Momentum négatif ({momentum:.1f}%)')
        else:
            return Signal('momentum', 0, self.weights['momentum'], 0.4, f'Momentum faible ({momentum:.1f}%)')
    
    def _evaluate_volatility(self, atr_pct: float, threshold: float = 2.0) -> Signal:
        """Évalue le signal de volatilité"""
        if pd.isna(atr_pct):
            return Signal('volatility', 0, self.weights['volatility'], 0, 'ATR non disponible')
        
        # Volatilité comme filtre
        if atr_pct < 0.5:
            return Signal('volatility', -0.5, self.weights['volatility'], 0.8, f'Volatilité trop faible ({atr_pct:.2f}%)')
        elif atr_pct > 5:
            return Signal('volatility', -0.3, self.weights['volatility'], 0.7, f'Volatilité excessive ({atr_pct:.2f}%)')
        else:
            # Volatilité idéale entre 1-3%
            quality = 1 - abs(atr_pct - 2) / 3
            return Signal('volatility', quality * 0.3, self.weights['volatility'], 0.6, f'Volatilité favorable ({atr_pct:.2f}%)')
    
    def calculate_score(self, df: pd.DataFrame) -> TradingScore:
        """Calcule le score de trading composite"""
        
        # S'assurer qu'on a assez de données
        if len(df) < 30:
            return TradingScore(
                total_score=0,
                direction='NEUTRAL',
                confidence=0,
                signals=[],
                timestamp=datetime.now()
            )
        
        # Calculer les indicateurs
        df = self.calculate_indicators(df)
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # Calculer le changement de prix
        price_change = (last_row['close'] - prev_row['close']) / prev_row['close'] * 100
        
        # Évaluer chaque signal
        signals = [
            self._evaluate_rsi(last_row['rsi']),
            self._evaluate_bollinger(last_row['bb_position'], last_row['close'], 
                                   last_row['bb_upper'], last_row['bb_lower']),
            self._evaluate_macd(last_row['macd'], last_row['macd_signal'], 
                              last_row['macd_histogram']),
            self._evaluate_volume(last_row['volume_ratio'], price_change),
            self._evaluate_ma_cross(last_row['ma_diff'], last_row['ma_fast'], 
                                  last_row['ma_slow']),
            self._evaluate_momentum(last_row['momentum']),
            self._evaluate_volatility(last_row['atr_pct'])
        ]
        
        # Calculer le score total
        total_score = sum(signal.weighted_value for signal in signals)
        
        # Calculer la confidence moyenne pondérée
        total_weight = sum(s.weight * s.confidence for s in signals)
        total_confidence = total_weight / sum(s.weight for s in signals) if signals else 0
        
        # Déterminer la direction
        if total_score >= self.strong_buy_threshold:
            direction = 'STRONG_BUY'
        elif total_score >= self.buy_threshold:
            direction = 'BUY'
        elif total_score <= self.strong_sell_threshold:
            direction = 'STRONG_SELL'
        elif total_score <= self.sell_threshold:
            direction = 'SELL'
        else:
            direction = 'NEUTRAL'
        
        # Créer le score final
        trading_score = TradingScore(
            total_score=total_score,
            direction=direction,
            confidence=total_confidence,
            signals=signals,
            timestamp=datetime.now()
        )
        
        # Logger les détails
        log_debug(f"Score calculé pour {self.symbol}: {direction} (score={total_score:.3f}, conf={total_confidence:.2f})")
        
        # Logger les signaux contributeurs
        if direction != 'NEUTRAL':
            top_signals = sorted(signals, key=lambda s: abs(s.weighted_value), reverse=True)[:3]
            for signal in top_signals:
                if abs(signal.weighted_value) > 0.05:
                    log_debug(f"  → {signal.name}: {signal.reason} (contribution: {signal.weighted_value:.3f})")
        
        return trading_score
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Met à jour les poids des indicateurs"""
        # Vérifier que la somme fait 1.0
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.001:
            # Normaliser
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        self.weights.update(new_weights)
        log_info(f"Poids mis à jour: {self.weights}")
    
    def get_decision_explanation(self, score: TradingScore) -> str:
        """Génère une explication lisible de la décision"""
        if score.direction == 'NEUTRAL':
            return "Pas de signal clair, rester en dehors du marché"
        
        # Trier les signaux par contribution
        sorted_signals = sorted(score.signals, 
                              key=lambda s: abs(s.weighted_value), 
                              reverse=True)
        
        # Prendre les 3 principaux contributeurs
        main_reasons = []
        for signal in sorted_signals[:3]:
            if abs(signal.weighted_value) > 0.05:
                main_reasons.append(signal.reason)
        
        direction_text = {
            'STRONG_BUY': "Signal d'achat très fort",
            'BUY': "Signal d'achat",
            'STRONG_SELL': "Signal de vente très fort",
            'SELL': "Signal de vente"
        }
        
        explanation = f"{direction_text.get(score.direction, score.direction)} " \
                     f"(score: {score.total_score:.2f}, confiance: {score.confidence:.0%})\n"
        
        if main_reasons:
            explanation += "Raisons principales:\n"
            for i, reason in enumerate(main_reasons, 1):
                explanation += f"  {i}. {reason}\n"
        
        return explanation