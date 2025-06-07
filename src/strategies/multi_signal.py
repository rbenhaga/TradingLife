"""
Stratégie de trading basée sur plusieurs signaux combinés
Plus agressive que la simple MA Cross pour générer plus de signaux
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.core.logger import log_info, log_debug, log_warning
from src.core.weighted_score_engine import WeightedScoreEngine, TradingScore

class MultiSignalStrategy:
    """Stratégie combinant plusieurs indicateurs pour des signaux plus fréquents"""
    
    def __init__(self, symbol: str = 'BTC/USDT'):
        """Initialise la stratégie multi-signaux avec score pondéré"""
        self.symbol = symbol
        self.data = pd.DataFrame()
        self.current_position = None
        self.entry_price = None
        self.last_signal_time = None
        
        # Moteur de score pondéré
        self.score_engine = WeightedScoreEngine(symbol)
        
        # Paramètres de trading
        self.stop_loss_pct = 0.015  # 1.5%
        self.take_profit_pct = 0.03  # 3%
        
        # Derniers scores pour historique
        self.last_score = None
        self.score_history = []
        
        log_info(f"Stratégie Multi-Signal avec score pondéré initialisée pour {symbol}")
        
    def update(self, klines: List[Dict]):
        """Met à jour les données et calcule le score"""
        # Convertir en DataFrame
        df = pd.DataFrame(klines)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Stocker les données
        self.data = df
        
        # Calculer le score avec le moteur
        if len(df) >= 30:  # Besoin d'au moins 30 bougies
            self.last_score = self.score_engine.calculate_score(df)
            
            # Ajouter à l'historique (garder les 100 derniers)
            self.score_history.append({
                'timestamp': self.last_score.timestamp,
                'score': self.last_score.total_score,
                'direction': self.last_score.direction,
                'confidence': self.last_score.confidence
            })
            if len(self.score_history) > 100:
                self.score_history.pop(0)
            
            # Log du score actuel
            if self.last_score.direction != 'NEUTRAL':
                log_debug(f"{self.symbol} - {self.score_engine.get_decision_explanation(self.last_score)}")
        else:
            log_debug(f"{self.symbol} - Pas assez de données ({len(df)} bougies)")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _get_signal_strength(self) -> Tuple[float, float]:
        """
        Calcule la force des signaux d'achat et de vente
        Returns: (buy_strength, sell_strength) entre 0 et 100
        """
        if len(self.data) < 30:
            return 0, 0
        
        last = self.data.iloc[-1]
        buy_signals = 0
        sell_signals = 0
        
        # 1. Signal RSI
        if pd.notna(last['rsi']):
            if last['rsi'] < self.rsi_oversold:
                buy_signals += 25
            elif last['rsi'] > self.rsi_overbought:
                sell_signals += 25
                
        # 2. Signal Bollinger Bands
        if pd.notna(last['bb_position']):
            if last['bb_position'] < 0.2:  # Proche de la bande inférieure
                buy_signals += 25
            elif last['bb_position'] > 0.8:  # Proche de la bande supérieure
                sell_signals += 25
                
        # 3. Signal MA Cross
        if pd.notna(last['ma_fast']) and pd.notna(last['ma_slow']):
            if last['ma_fast'] > last['ma_slow']:
                buy_signals += 15
            else:
                sell_signals += 15
                
        # 4. Signal Momentum
        if pd.notna(last['momentum']):
            if last['momentum'] < -1:  # Momentum négatif fort
                buy_signals += 20
            elif last['momentum'] > 1:  # Momentum positif fort
                sell_signals += 20
                
        # 5. Signal Volume
        if pd.notna(last['volume_ratio']):
            if last['volume_ratio'] > 1.5:  # Volume élevé
                buy_signals += 15
                sell_signals += 15
                
        return buy_signals, sell_signals
    
    def get_signal(self) -> Optional[str]:
        """Retourne le signal de trading basé sur le score pondéré"""
        if not self.last_score or len(self.data) < 30:
            log_debug("Pas assez de données pour générer un signal")
            return None
        
        last = self.data.iloc[-1]
        current_price = last['close']
        
        # Vérifier stop loss/take profit si en position
        if self.current_position == 'LONG' and self.entry_price:
            stop_loss = self.entry_price * (1 - self.stop_loss_pct)
            take_profit = self.entry_price * (1 + self.take_profit_pct)
            
            if current_price <= stop_loss:
                log_info(f"🛑 Stop loss touché à {current_price:.2f}")
                return 'SELL'
            elif current_price >= take_profit:
                log_info(f"🎯 Take profit touché à {current_price:.2f}")
                return 'SELL'
        
        # Utiliser la décision du score engine
        if self.current_position is None:
            # Recherche d'entrée
            if self.last_score.direction in ['BUY', 'STRONG_BUY']:
                explanation = self.score_engine.get_decision_explanation(self.last_score)
                log_info(f"📈 {self.symbol} - {explanation}")
                self.entry_price = current_price
                return 'BUY'
        else:
            # Recherche de sortie
            if self.last_score.direction in ['SELL', 'STRONG_SELL']:
                explanation = self.score_engine.get_decision_explanation(self.last_score)
                log_info(f"📉 {self.symbol} - {explanation}")
                return 'SELL'
            
            # Sortie si le signal devient neutre avec perte
            if self.last_score.direction == 'NEUTRAL':
                pnl_pct = ((current_price / self.entry_price) - 1) * 100
                if pnl_pct < -1.0:  # Perte > 1%
                    log_info(f"⚠️ {self.symbol} - Signal neutre avec perte, sortie préventive")
                    return 'SELL'
        
        return None
    
    def get_current_price(self) -> float:
        """Retourne le prix actuel"""
        if len(self.data) == 0:
            return 0.0
        return float(self.data['close'].iloc[-1])
    
    def get_indicators(self) -> Dict:
        """Retourne tous les indicateurs actuels avec le score"""
        if len(self.data) == 0:
            return {}
        
        # Indicateurs de base
        indicators = {
            'current_price': float(self.data['close'].iloc[-1]),
            'symbol': self.symbol
        }
        
        # Ajouter les infos du score si disponible
        if self.last_score:
            indicators.update({
                'total_score': self.last_score.total_score,
                'direction': self.last_score.direction,
                'confidence': self.last_score.confidence,
                'buy_strength': max(0, self.last_score.total_score * 100),
                'sell_strength': max(0, -self.last_score.total_score * 100)
            })
            
            # Ajouter les top 3 signaux contributeurs
            top_signals = sorted(self.last_score.signals, 
                               key=lambda s: abs(s.weighted_value), 
                               reverse=True)[:3]
            
            indicators['top_signals'] = [
                {
                    'name': s.name,
                    'contribution': s.weighted_value,
                    'reason': s.reason
                }
                for s in top_signals if abs(s.weighted_value) > 0.05
            ]
        
        return indicators
    
    def get_performance_metrics(self) -> Dict:
        """Retourne les métriques de performance avec historique de score"""
        if len(self.data) < 2:
            return {}
        
        metrics = {}
        
        # Métriques de base
        if len(self.score_history) > 0:
            recent_scores = [s['score'] for s in self.score_history[-20:]]
            metrics['avg_score'] = np.mean(recent_scores)
            metrics['score_volatility'] = np.std(recent_scores)
            
            # Tendance du score (régression linéaire simple)
            if len(recent_scores) > 5:
                x = np.arange(len(recent_scores))
                slope = np.polyfit(x, recent_scores, 1)[0]
                metrics['score_trend'] = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
        
        return metrics