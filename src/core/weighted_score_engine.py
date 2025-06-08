"""
Moteur de score pondéré pour l'analyse multi-signal
Version corrigée avec TradingScore dataclass
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger('WeightedScoreEngine')

@dataclass
class SignalComponent:
    """Composant individuel d'un signal"""
    name: str
    value: float  # Valeur brute du signal (-1 à 1)
    weight: float  # Poids du signal (0 à 1)
    weighted_value: float  # Valeur pondérée
    confidence: float  # Niveau de confiance (0 à 1)
    reason: str  # Explication du signal

@dataclass
class TradingScore:
    """Score de trading complet avec tous les détails"""
    symbol: str
    timestamp: datetime
    total_score: float  # Score final pondéré (-1 à 1)
    confidence: float  # Confiance globale (0 à 1)
    direction: str  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    signals: List[SignalComponent] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def get_action(self) -> str:
        """Retourne l'action recommandée basée sur le score"""
        if self.total_score > 0.7:
            return 'STRONG_BUY'
        elif self.total_score > 0.3:
            return 'BUY'
        elif self.total_score < -0.7:
            return 'STRONG_SELL'
        elif self.total_score < -0.3:
            return 'SELL'
        else:
            return 'NEUTRAL'

class WeightedScoreEngine:
    """
    Moteur de calcul de score pondéré pour combiner plusieurs indicateurs
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialise le moteur de score
        
        Args:
            weights: Poids personnalisés pour chaque indicateur
        """
        # Poids par défaut équilibrés
        self.default_weights = {
            'rsi': 0.20,
            'bollinger': 0.20,
            'macd': 0.15,
            'volume': 0.15,
            'ma_cross': 0.10,
            'momentum': 0.10,
            'volatility': 0.10
        }
        
        # Utiliser les poids personnalisés ou les poids par défaut
        self.weights = weights or self.default_weights.copy()
        
        # Normaliser les poids pour qu'ils somment à 1
        self._normalize_weights()
        
        logger.info(f"Score Engine initialisé avec poids: {self.weights}")
    
    def _normalize_weights(self):
        """Normalise les poids pour qu'ils somment à 1"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def calculate_score(self, signals: Dict[str, Dict], symbol: str = "UNKNOWN") -> TradingScore:
        """
        Calcule le score pondéré à partir des signaux
        
        Args:
            signals: Dict des signaux par indicateur
                    Format: {'indicator': {'signal': float, 'confidence': float, 'reason': str}}
            symbol: Symbole tradé
        
        Returns:
            TradingScore avec tous les détails
        """
        if not signals:
            return TradingScore(
                symbol=symbol,
                timestamp=datetime.now(),
                total_score=0.0,
                confidence=0.0,
                direction='NEUTRAL',
                signals=[],
                metadata={'empty_signals': True}
            )
        
        signal_components = []
        total_score = 0.0
        total_confidence = 0.0
        weights_sum = 0.0
        
        # Calculer le score pondéré
        for indicator, weight in self.weights.items():
            if indicator in signals:
                signal_data = signals[indicator]
                signal_value = float(signal_data.get('signal', 0))
                confidence = float(signal_data.get('confidence', 0.5))
                reason = signal_data.get('reason', f'{indicator} signal')
                
                # Valider et limiter les valeurs
                signal_value = np.clip(signal_value, -1, 1)
                confidence = np.clip(confidence, 0, 1)
                
                # Contribution au score total
                weighted_value = signal_value * weight
                total_score += weighted_value
                
                # Moyenne pondérée de la confiance
                total_confidence += confidence * weight
                weights_sum += weight
                
                # Créer le composant de signal
                component = SignalComponent(
                    name=indicator,
                    value=signal_value,
                    weight=weight,
                    weighted_value=weighted_value,
                    confidence=confidence,
                    reason=reason
                )
                signal_components.append(component)
        
        # Normaliser la confiance si nécessaire
        if weights_sum > 0:
            total_confidence = total_confidence / weights_sum
        
        # Créer le score de trading
        trading_score = TradingScore(
            symbol=symbol,
            timestamp=datetime.now(),
            total_score=float(np.clip(total_score, -1, 1)),
            confidence=float(total_confidence),
            direction='',  # Sera défini après
            signals=signal_components,
            metadata={
                'indicators_used': len(signal_components),
                'weights_sum': weights_sum
            }
        )
        
        # Définir la direction
        trading_score.direction = trading_score.get_action()
        
        return trading_score
    
    def analyze_indicators(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict[str, Dict]:
        """
        Analyse les indicateurs techniques et génère les signaux
        
        Args:
            df: DataFrame avec les données de marché
            symbol: Symbole tradé
        
        Returns:
            Dict des signaux par indicateur
        """
        signals = {}
        
        if len(df) < 20:
            logger.warning("Pas assez de données pour l'analyse")
            return signals
        
        # RSI Signal
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if pd.notna(rsi):
                if rsi < 30:
                    signals['rsi'] = {
                        'signal': 1.0, 
                        'confidence': 0.9,
                        'reason': f'RSI oversold ({rsi:.1f})'
                    }
                elif rsi > 70:
                    signals['rsi'] = {
                        'signal': -1.0, 
                        'confidence': 0.9,
                        'reason': f'RSI overbought ({rsi:.1f})'
                    }
                else:
                    # Signal proportionnel
                    signal = (50 - rsi) / 50  # Positif si RSI < 50
                    signals['rsi'] = {
                        'signal': signal, 
                        'confidence': 0.5,
                        'reason': f'RSI neutral ({rsi:.1f})'
                    }
        
        # Bollinger Bands Signal
        if all(col in df.columns for col in ['close', 'bb_lower', 'bb_upper']):
            close = df['close'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_middle = df['bb_middle'].iloc[-1] if 'bb_middle' in df.columns else (bb_lower + bb_upper) / 2
            
            if close < bb_lower:
                signals['bollinger'] = {
                    'signal': 1.0, 
                    'confidence': 0.8,
                    'reason': 'Price below lower band'
                }
            elif close > bb_upper:
                signals['bollinger'] = {
                    'signal': -1.0, 
                    'confidence': 0.8,
                    'reason': 'Price above upper band'
                }
            else:
                # Position relative dans les bandes
                if bb_upper > bb_lower:
                    position = (close - bb_lower) / (bb_upper - bb_lower)
                    signal = 1 - 2 * position  # +1 en bas, -1 en haut
                    signals['bollinger'] = {
                        'signal': signal, 
                        'confidence': 0.5,
                        'reason': f'Price at {position*100:.0f}% of bands'
                    }
        
        # MACD Signal
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_prev = df['macd'].iloc[-2] if len(df) > 1 else macd
            signal_prev = df['macd_signal'].iloc[-2] if len(df) > 1 else macd_signal
            
            # Croisement MACD
            if macd > macd_signal and macd_prev <= signal_prev:
                signals['macd'] = {
                    'signal': 1.0, 
                    'confidence': 0.85,
                    'reason': 'MACD bullish crossover'
                }
            elif macd < macd_signal and macd_prev >= signal_prev:
                signals['macd'] = {
                    'signal': -1.0, 
                    'confidence': 0.85,
                    'reason': 'MACD bearish crossover'
                }
            else:
                # Distance relative
                diff = (macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 0
                signals['macd'] = {
                    'signal': np.clip(diff * 2, -1, 1), 
                    'confidence': 0.6,
                    'reason': f'MACD {"above" if diff > 0 else "below"} signal'
                }
        
        # Volume Signal
        if 'volume' in df.columns and len(df) >= 20:
            recent_volume = df['volume'].iloc[-5:].mean()
            avg_volume = df['volume'].iloc[-20:].mean()
            
            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                if volume_ratio > 1.5:
                    # Volume élevé = confirmation de tendance
                    price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                    signals['volume'] = {
                        'signal': np.sign(price_change) * 0.8,
                        'confidence': 0.7,
                        'reason': f'High volume ({volume_ratio:.1f}x avg) {"up" if price_change > 0 else "down"} move'
                    }
                else:
                    signals['volume'] = {
                        'signal': 0.0, 
                        'confidence': 0.3,
                        'reason': f'Normal volume ({volume_ratio:.1f}x avg)'
                    }
        
        # MA Cross Signal
        if all(col in df.columns for col in ['ma_fast', 'ma_slow']):
            ma_fast = df['ma_fast'].iloc[-1]
            ma_slow = df['ma_slow'].iloc[-1]
            ma_fast_prev = df['ma_fast'].iloc[-2] if len(df) > 1 else ma_fast
            ma_slow_prev = df['ma_slow'].iloc[-2] if len(df) > 1 else ma_slow
            
            # Croisement des moyennes mobiles
            if ma_fast > ma_slow and ma_fast_prev <= ma_slow_prev:
                signals['ma_cross'] = {
                    'signal': 1.0, 
                    'confidence': 0.75,
                    'reason': 'Golden cross (bullish MA crossover)'
                }
            elif ma_fast < ma_slow and ma_fast_prev >= ma_slow_prev:
                signals['ma_cross'] = {
                    'signal': -1.0, 
                    'confidence': 0.75,
                    'reason': 'Death cross (bearish MA crossover)'
                }
            else:
                # Distance relative
                diff = (ma_fast - ma_slow) / ma_slow if ma_slow != 0 else 0
                signals['ma_cross'] = {
                    'signal': np.clip(diff * 10, -1, 1), 
                    'confidence': 0.5,
                    'reason': f'Fast MA {"above" if diff > 0 else "below"} slow MA by {abs(diff)*100:.1f}%'
                }
        
        # Momentum Signal
        if 'close' in df.columns and len(df) >= 10:
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            signals['momentum'] = {
                'signal': np.clip(momentum * 10, -1, 1),
                'confidence': 0.6,
                'reason': f'{abs(momentum)*100:.1f}% {"gain" if momentum > 0 else "loss"} over 10 periods'
            }
        
        # Volatility Signal
        if 'close' in df.columns and len(df) >= 20:
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 20:
                volatility = returns.iloc[-20:].std()
                avg_volatility = returns.std()
                
                if avg_volatility > 0:
                    vol_ratio = volatility / avg_volatility
                    if 0.5 < vol_ratio < 2:
                        # Volatilité idéale pour le trading
                        signals['volatility'] = {
                            'signal': 0.5, 
                            'confidence': 0.7,
                            'reason': f'Good volatility ({vol_ratio:.1f}x average)'
                        }
                    elif vol_ratio > 2:
                        # Trop volatil = risqué
                        signals['volatility'] = {
                            'signal': -0.3, 
                            'confidence': 0.8,
                            'reason': f'High volatility ({vol_ratio:.1f}x average)'
                        }
                    else:
                        # Pas assez volatil
                        signals['volatility'] = {
                            'signal': -0.1, 
                            'confidence': 0.5,
                            'reason': f'Low volatility ({vol_ratio:.1f}x average)'
                        }
        
        return signals
    
    def get_visual_score(self, trading_score: TradingScore) -> str:
        """
        Retourne une représentation visuelle du score
        
        Args:
            trading_score: Score de trading
        
        Returns:
            String avec représentation visuelle
        """
        # Barre de progression
        bar_length = 20
        filled = int((trading_score.total_score + 1) * bar_length / 2)
        
        if trading_score.total_score > 0.5:
            color = '🟢'
        elif trading_score.total_score > 0:
            color = '🟡'
        elif trading_score.total_score > -0.5:
            color = '🟠'
        else:
            color = '🔴'
        
        bar = color + ' [' + '=' * filled + ' ' * (bar_length - filled) + ']'
        
        # Texte de confiance
        if trading_score.confidence > 0.8:
            conf_text = "Très confiant"
        elif trading_score.confidence > 0.6:
            conf_text = "Confiant"
        elif trading_score.confidence > 0.4:
            conf_text = "Moyennement confiant"
        else:
            conf_text = "Peu confiant"
        
        return f"{bar} Score: {trading_score.total_score:.3f} ({conf_text}: {trading_score.confidence:.1%})"
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Met à jour les poids des indicateurs
        
        Args:
            new_weights: Nouveaux poids
        """
        self.weights = new_weights.copy()
        self._normalize_weights()
        logger.info(f"Poids mis à jour: {self.weights}")