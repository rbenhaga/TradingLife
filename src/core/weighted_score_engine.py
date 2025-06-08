"""
Moteur de score pondéré pour l'analyse multi-signal
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

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
        self.weights = weights or self.default_weights
        
        # Normaliser les poids pour qu'ils somment à 1
        self._normalize_weights()
        
        self.logger = logging.getLogger('WeightedScoreEngine')
        self.logger.info(f"Score Engine initialisé avec poids: {self.weights}")
    
    def _normalize_weights(self):
        """Normalise les poids pour qu'ils somment à 1"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def calculate_score(self, signals: Dict[str, Dict]) -> Dict:
        """
        Calcule le score pondéré à partir des signaux
        
        Args:
            signals: Dictionnaire des signaux par indicateur
                    Format: {'indicator': {'signal': float, 'confidence': float}}
        
        Returns:
            Dict avec score total, confidence et détails
        """
        if not signals:
            return {
                'score': 0.0,
                'confidence': 0.0,
                'details': {},
                'action': 'NEUTRAL'
            }
        
        total_score = 0.0
        total_confidence = 0.0
        details = {}
        
        # Calculer le score pondéré
        for indicator, weight in self.weights.items():
            if indicator in signals:
                signal_data = signals[indicator]
                signal_value = signal_data.get('signal', 0)
                confidence = signal_data.get('confidence', 0)
                
                # Contribution au score total
                contribution = signal_value * weight
                total_score += contribution
                
                # Moyenne pondérée de la confiance
                total_confidence += confidence * weight
                
                # Détails pour le debug
                details[indicator] = {
                    'signal': signal_value,
                    'confidence': confidence,
                    'weight': weight,
                    'contribution': contribution
                }
        
        # Déterminer l'action basée sur le score
        if total_score > 0.5:
            action = 'STRONG_BUY'
        elif total_score > 0.3:
            action = 'BUY'
        elif total_score < -0.5:
            action = 'STRONG_SELL'
        elif total_score < -0.3:
            action = 'SELL'
        else:
            action = 'NEUTRAL'
        
        return {
            'score': total_score,
            'confidence': total_confidence,
            'details': details,
            'action': action
        }
    
    def analyze_indicators(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyse les indicateurs techniques et génère les signaux
        
        Args:
            df: DataFrame avec les données de marché
        
        Returns:
            Dict des signaux par indicateur
        """
        signals = {}
        
        if len(df) < 20:
            self.logger.warning("Pas assez de données pour l'analyse")
            return signals
        
        # RSI Signal
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if pd.notna(rsi):
                if rsi < 30:
                    signals['rsi'] = {'signal': 1.0, 'confidence': 0.9}
                elif rsi > 70:
                    signals['rsi'] = {'signal': -1.0, 'confidence': 0.9}
                else:
                    # Signal proportionnel
                    signal = (50 - rsi) / 50  # Positif si RSI < 50
                    signals['rsi'] = {'signal': signal, 'confidence': 0.5}
        
        # Bollinger Bands Signal
        if all(col in df.columns for col in ['close', 'bb_lower', 'bb_upper']):
            close = df['close'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_middle = df['bb_middle'].iloc[-1] if 'bb_middle' in df.columns else (bb_lower + bb_upper) / 2
            
            if close < bb_lower:
                signals['bollinger'] = {'signal': 1.0, 'confidence': 0.8}
            elif close > bb_upper:
                signals['bollinger'] = {'signal': -1.0, 'confidence': 0.8}
            else:
                # Position relative dans les bandes
                position = (close - bb_lower) / (bb_upper - bb_lower)
                signal = 1 - 2 * position  # +1 en bas, -1 en haut
                signals['bollinger'] = {'signal': signal, 'confidence': 0.5}
        
        # MACD Signal
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_prev = df['macd'].iloc[-2]
            signal_prev = df['macd_signal'].iloc[-2]
            
            # Croisement MACD
            if macd > macd_signal and macd_prev <= signal_prev:
                signals['macd'] = {'signal': 1.0, 'confidence': 0.85}
            elif macd < macd_signal and macd_prev >= signal_prev:
                signals['macd'] = {'signal': -1.0, 'confidence': 0.85}
            else:
                # Distance relative
                diff = (macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 0
                signals['macd'] = {'signal': np.clip(diff * 2, -1, 1), 'confidence': 0.6}
        
        # Volume Signal
        if 'volume' in df.columns:
            recent_volume = df['volume'].iloc[-5:].mean()
            avg_volume = df['volume'].iloc[-20:].mean()
            
            if recent_volume > avg_volume * 1.5:
                # Volume élevé = confirmation de tendance
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                signals['volume'] = {
                    'signal': np.sign(price_change) * 0.8,
                    'confidence': 0.7
                }
            else:
                signals['volume'] = {'signal': 0.0, 'confidence': 0.3}
        
        # MA Cross Signal
        if all(col in df.columns for col in ['ma_fast', 'ma_slow']):
            ma_fast = df['ma_fast'].iloc[-1]
            ma_slow = df['ma_slow'].iloc[-1]
            ma_fast_prev = df['ma_fast'].iloc[-2]
            ma_slow_prev = df['ma_slow'].iloc[-2]
            
            # Croisement des moyennes mobiles
            if ma_fast > ma_slow and ma_fast_prev <= ma_slow_prev:
                signals['ma_cross'] = {'signal': 1.0, 'confidence': 0.75}
            elif ma_fast < ma_slow and ma_fast_prev >= ma_slow_prev:
                signals['ma_cross'] = {'signal': -1.0, 'confidence': 0.75}
            else:
                # Distance relative
                diff = (ma_fast - ma_slow) / ma_slow if ma_slow != 0 else 0
                signals['ma_cross'] = {'signal': np.clip(diff * 10, -1, 1), 'confidence': 0.5}
        
        # Momentum Signal
        if 'close' in df.columns and len(df) >= 10:
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            signals['momentum'] = {
                'signal': np.clip(momentum * 10, -1, 1),
                'confidence': 0.6
            }
        
        # Volatility Signal (préfère les marchés volatiles mais pas trop)
        if 'close' in df.columns and len(df) >= 20:
            returns = df['close'].pct_change().dropna()
            volatility = returns.iloc[-20:].std()
            avg_volatility = returns.std()
            
            if 0.5 * avg_volatility < volatility < 2 * avg_volatility:
                # Volatilité idéale pour le trading
                signals['volatility'] = {'signal': 0.5, 'confidence': 0.7}
            elif volatility > 2 * avg_volatility:
                # Trop volatil = risqué
                signals['volatility'] = {'signal': -0.3, 'confidence': 0.8}
            else:
                # Pas assez volatil
                signals['volatility'] = {'signal': -0.1, 'confidence': 0.5}
        
        return signals
    
    def get_visual_score(self, score: float, confidence: float) -> str:
        """
        Retourne une représentation visuelle du score
        
        Args:
            score: Score de -1 à 1
            confidence: Confiance de 0 à 1
        
        Returns:
            String avec représentation visuelle
        """
        # Barre de progression
        bar_length = 20
        filled = int((score + 1) * bar_length / 2)
        
        if score > 0.5:
            color = '🟢'
        elif score > 0:
            color = '🟡'
        elif score > -0.5:
            color = '🟠'
        else:
            color = '🔴'
        
        bar = color + ' [' + '=' * filled + ' ' * (bar_length - filled) + ']'
        
        # Texte de confiance
        if confidence > 0.8:
            conf_text = "Très confiant"
        elif confidence > 0.6:
            conf_text = "Confiant"
        elif confidence > 0.4:
            conf_text = "Moyennement confiant"
        else:
            conf_text = "Peu confiant"
        
        return f"{bar} Score: {score:.3f} ({conf_text}: {confidence:.1%})"
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Met à jour les poids des indicateurs
        
        Args:
            new_weights: Nouveaux poids
        """
        self.weights = new_weights
        self._normalize_weights()
        self.logger.info(f"Poids mis à jour: {self.weights}")