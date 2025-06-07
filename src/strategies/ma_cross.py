"""
Stratégie de trading basée sur le croisement des moyennes mobiles
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.core.logger import log_info, log_debug, log_warning

class MovingAverageCrossStrategy:
    """Stratégie de trading basée sur le croisement des moyennes mobiles"""
    
    def __init__(self, short_window: int = 5, long_window: int = 13, trend_window: int = 50, 
                 stop_loss_pct: float = 0.02, take_profit_pct: float = 0.05, symbol: str = 'BTC/USDT'):
        """
        Initialise la stratégie
        
        Args:
            short_window: Période de la moyenne mobile courte
            long_window: Période de la moyenne mobile longue
            trend_window: Période de la moyenne mobile de tendance
            stop_loss_pct: Pourcentage de stop loss
            take_profit_pct: Pourcentage de take profit
            symbol: Paire de trading
        """
        self.short_window = short_window
        self.long_window = long_window
        self.trend_window = trend_window
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.symbol = symbol
        self.data = pd.DataFrame()
        self.current_position = None
        self.last_signal = None
        self.entry_price = None
        self.last_cross_time = None
        self.min_bars_between_trades = 3  # Éviter les faux signaux
        
        log_info(f"Stratégie initialisée - MA courte: {short_window}, MA longue: {long_window}, MA tendance: {trend_window}")
        
    def update(self, klines: List[Dict]):
        """
        Met à jour les données et calcule les indicateurs
        
        Args:
            klines: Liste des bougies OHLCV
        """
        # Convertir les klines en DataFrame
        df = pd.DataFrame(klines)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Calculer les moyennes mobiles
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        df['trend_ma'] = df['close'].rolling(window=self.trend_window).mean()
        
        # Calculer le RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Calculer la volatilité
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
        
        # Détection des croisements
        df['ma_diff'] = df['short_ma'] - df['long_ma']
        df['ma_diff_prev'] = df['ma_diff'].shift(1)
        
        # Signaux de croisement
        df['cross_up'] = (df['ma_diff'] > 0) & (df['ma_diff_prev'] <= 0)
        df['cross_down'] = (df['ma_diff'] < 0) & (df['ma_diff_prev'] >= 0)
        
        # Stocker les données
        self.data = df
        
        # Log des dernières valeurs pour debug
        if not df.empty and len(df) > 0:
            last_row = df.iloc[-1]
            log_debug(f"Prix actuel: {last_row['close']:.2f} USDT")
            log_debug(f"MA courte ({self.short_window}): {last_row['short_ma']:.2f} USDT")
            log_debug(f"MA longue ({self.long_window}): {last_row['long_ma']:.2f} USDT")
            if pd.notna(last_row['trend_ma']):
                log_debug(f"MA tendance ({self.trend_window}): {last_row['trend_ma']:.2f} USDT")
            if pd.notna(last_row['rsi']):
                log_debug(f"RSI: {last_row['rsi']:.2f}")
    
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
    
    def get_signal(self) -> Optional[str]:
        """
        Retourne le signal de trading actuel
        
        Returns:
            'BUY' pour un signal d'achat
            'SELL' pour un signal de vente
            None si pas de signal
        """
        if len(self.data) < self.long_window + 5:
            log_debug("Pas assez de données pour générer un signal")
            return None
        
        # Obtenir les dernières lignes
        last_row = self.data.iloc[-1]
        current_price = last_row['close']
        
        # Vérifier le stop loss/take profit si en position
        if self.current_position == 'LONG' and self.entry_price is not None:
            stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
            take_profit_price = self.entry_price * (1 + self.take_profit_pct)
            
            if current_price <= stop_loss_price:
                log_info(f"🛑 Stop loss atteint - Prix: {current_price:.2f}, Stop: {stop_loss_price:.2f}")
                return 'SELL'
            elif current_price >= take_profit_price:
                log_info(f"🎯 Take profit atteint - Prix: {current_price:.2f}, TP: {take_profit_price:.2f}")
                return 'SELL'
        
        # Vérifier s'il y a eu un croisement récent
        recent_data = self.data.tail(self.min_bars_between_trades + 1)
        
        # Signal d'achat : croisement haussier
        if any(recent_data['cross_up']) and self.current_position is None:
            # Vérifications supplémentaires pour filtrer les faux signaux
            
            # 1. Vérifier la tendance si disponible
            if pd.notna(last_row['trend_ma']):
                if current_price < last_row['trend_ma']:
                    log_debug("Signal d'achat ignoré - Prix sous la MA de tendance")
                    return None
            
            # 2. Vérifier le RSI (éviter les surachats)
            if pd.notna(last_row['rsi']) and last_row['rsi'] > 70:
                log_debug("Signal d'achat ignoré - RSI en surachat")
                return None
            
            # 3. Vérifier la volatilité (éviter les marchés trop calmes)
            if pd.notna(last_row['volatility']) and last_row['volatility'] < 0.1:
                log_debug("Signal d'achat ignoré - Volatilité trop faible")
                return None
            
            log_info(f"📈 Signal d'ACHAT détecté - Croisement haussier MA{self.short_window} > MA{self.long_window}")
            self.entry_price = current_price
            return 'BUY'
        
        # Signal de vente : croisement baissier (si en position)
        elif any(recent_data['cross_down']) and self.current_position == 'LONG':
            log_info(f"📉 Signal de VENTE détecté - Croisement baissier MA{self.short_window} < MA{self.long_window}")
            return 'SELL'
        
        return None
    
    def get_current_price(self) -> float:
        """Retourne le prix actuel"""
        if len(self.data) == 0:
            return 0.0
        return float(self.data['close'].iloc[-1])
    
    def get_indicators(self) -> Dict:
        """Retourne les indicateurs actuels"""
        if len(self.data) == 0:
            return {
                'current_price': 0.0,
                'short_ma': 0.0,
                'long_ma': 0.0,
                'trend_ma': 0.0,
                'rsi': 0.0,
                'volatility': 0.0
            }
        
        last_row = self.data.iloc[-1]
        return {
            'current_price': float(last_row['close']),
            'short_ma': float(last_row['short_ma']) if pd.notna(last_row['short_ma']) else 0.0,
            'long_ma': float(last_row['long_ma']) if pd.notna(last_row['long_ma']) else 0.0,
            'trend_ma': float(last_row['trend_ma']) if pd.notna(last_row['trend_ma']) else 0.0,
            'rsi': float(last_row['rsi']) if pd.notna(last_row['rsi']) else 0.0,
            'volatility': float(last_row['volatility']) if pd.notna(last_row['volatility']) else 0.0
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calcule et retourne les métriques de performance"""
        if len(self.data) < 2:
            return {}
        
        # Calculer quelques métriques utiles
        returns = self.data['close'].pct_change().dropna()
        
        return {
            'current_drawdown': self._calculate_drawdown(),
            'volatility_24h': returns.tail(1440).std() * 100 if len(returns) > 1440 else 0,  # 24h pour 1m timeframe
            'price_change_24h': ((self.data['close'].iloc[-1] / self.data['close'].iloc[0]) - 1) * 100 if len(self.data) > 0 else 0
        }
    
    def _calculate_drawdown(self) -> float:
        """Calcule le drawdown actuel"""
        if len(self.data) < 2:
            return 0.0
        
        # Calculer le plus haut historique
        rolling_max = self.data['close'].expanding().max()
        drawdown = (self.data['close'] - rolling_max) / rolling_max * 100
        
        return float(drawdown.iloc[-1]) if len(drawdown) > 0 else 0.0