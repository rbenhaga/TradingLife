"""
Module de gestion des données de marché
Collecte, stocke et fournit les données OHLCV multi-timeframes
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.core.logger import log_info, log_debug, log_error
from ..exchanges.exchange_connector import ExchangeConnector

@dataclass
class MarketSnapshot:
    """Snapshot du marché à un instant T"""
    timestamp: datetime
    btc_price: float
    btc_change_24h: float
    total_market_cap: float
    fear_greed_index: int
    dominance_btc: float
    volume_24h: float
    top_gainers: List[str]
    top_losers: List[str]

class MarketData:
    """
    Gestionnaire centralisé des données de marché
    """
    
    def __init__(self, exchange_connector: ExchangeConnector, config: dict):
        """
        Initialise le gestionnaire de données
        
        Args:
            exchange_connector: Connecteur d'exchange
            config: Configuration du bot
        """
        self.exchange = exchange_connector
        self.config = config
        
        # Timeframes supportés
        self.timeframes = config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        
        # Données par paire et timeframe
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Dernière mise à jour
        self.last_update: Dict[str, datetime] = {}
        
        # Cache de tickers
        self.ticker_cache = {}
        self.ticker_cache_time = {}
        self.ticker_cache_duration = 10  # secondes
        
        # État du marché global
        self.market_snapshot = None
        self.last_snapshot_time = None
        
        # Configuration
        self.cache_size = self.config.get('cache_size', 1000)  # Bougies par timeframe
        self.update_interval = self.config.get('update_interval', 60)  # secondes
        
        log_info(f"MarketData initialisé - Timeframes: {self.timeframes}")
    
    async def initialize(self, pairs: List[str]):
        """
        Initialise les données pour les paires spécifiées
        
        Args:
            pairs: Liste des paires à initialiser
        """
        for pair in pairs:
            self.data[pair] = {}
            for tf in self.timeframes:
                await self.update_ohlcv(pair, tf)
    
    async def update_all(self):
        """Met à jour les données pour toutes les paires et timeframes"""
        try:
            for pair in self.data.keys():
                for tf in self.timeframes:
                    await self.update_ohlcv(pair, tf)
                    
            log_debug("Données de marché mises à jour")
            
        except Exception as e:
            log_error(f"Erreur lors de la mise à jour des données: {str(e)}")
    
    async def update_ohlcv(self, symbol: str, timeframe: str):
        """
        Met à jour les données OHLCV pour une paire et un timeframe
        
        Args:
            symbol: Symbole de la paire
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        """
        try:
            # Vérifier si une mise à jour est nécessaire
            last = self.last_update.get(f"{symbol}_{timeframe}")
            if last and datetime.now() - last < self._get_update_interval(timeframe):
                return
            
            # Récupérer les données
            ohlcv = await self.exchange.get_ohlcv(symbol, timeframe)
            if not ohlcv:
                return
            
            # Convertir en DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Mettre à jour les données
            self.data[symbol][timeframe] = df
            self.last_update[f"{symbol}_{timeframe}"] = datetime.now()
            
        except Exception as e:
            log_error(f"Erreur mise à jour OHLCV {symbol} {timeframe}: {str(e)}")
    
    def _get_update_interval(self, timeframe: str) -> timedelta:
        """Retourne l'intervalle de mise à jour pour un timeframe"""
        intervals = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        return intervals.get(timeframe, timedelta(minutes=5))
    
    def get_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Récupère les données pour une paire et un timeframe
        
        Args:
            symbol: Symbole de la paire
            timeframe: Timeframe
            
        Returns:
            DataFrame avec les données ou None si non disponible
        """
        return self.data.get(symbol, {}).get(timeframe)
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Récupère le ticker avec cache
        
        Args:
            symbol: Symbole
            
        Returns:
            Ticker data ou None
        """
        # Vérifier le cache
        if symbol in self.ticker_cache:
            cache_age = datetime.now() - self.ticker_cache_time[symbol]
            if cache_age.total_seconds() < self.ticker_cache_duration:
                return self.ticker_cache[symbol]
        
        # Récupérer depuis l'exchange
        ticker = await self.exchange.get_ticker(symbol)
        
        if ticker:
            self.ticker_cache[symbol] = ticker
            self.ticker_cache_time[symbol] = datetime.now()
        
        return ticker
    
    def calculate_indicators(self, symbol: str, timeframe: str = None) -> Dict:
        """
        Calcule les indicateurs techniques
        
        Args:
            symbol: Symbole
            timeframe: Timeframe (défaut: 15m)
            
        Returns:
            Dict avec tous les indicateurs
        """
        timeframe = timeframe or self.timeframes[0]
        df = self.get_data(symbol, timeframe)
        
        if len(df) < 50:
            return {}
        
        indicators = {}
        
        # Prix et volume actuels
        indicators['current_price'] = float(df['close'].iloc[-1])
        indicators['volume'] = float(df['volume'].iloc[-1])
        
        # Moyennes mobiles
        indicators['sma_20'] = float(df['close'].rolling(20).mean().iloc[-1])
        indicators['sma_50'] = float(df['close'].rolling(50).mean().iloc[-1])
        indicators['ema_12'] = float(df['close'].ewm(span=12).mean().iloc[-1])
        indicators['ema_26'] = float(df['close'].ewm(span=26).mean().iloc[-1])
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        macd_line = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = float(macd_line.iloc[-1])
        indicators['macd_signal'] = float(signal_line.iloc[-1])
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Bollinger Bands
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        indicators['bb_upper'] = float(sma.iloc[-1] + 2 * std.iloc[-1])
        indicators['bb_middle'] = float(sma.iloc[-1])
        indicators['bb_lower'] = float(sma.iloc[-1] - 2 * std.iloc[-1])
        
        # ATR (Average True Range)
        indicators['atr'] = self._calculate_atr(df)
        
        # Support et Résistance
        indicators['support'], indicators['resistance'] = self._calculate_support_resistance(df)
        
        # Momentum
        indicators['momentum_10'] = float((df['close'].iloc[-1] / df['close'].iloc[-11] - 1) * 100)
        
        # Force relative du volume
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = float(df['volume'].iloc[-1] / avg_volume) if avg_volume > 0 else 1
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        if loss.iloc[-1] == 0:
            return 100.0
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcule l'Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return float(atr.iloc[-1])
    
    def _calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float]:
        """Calcule les niveaux de support et résistance"""
        recent_data = df.tail(lookback)
        
        # Méthode simple : utiliser les min/max récents
        support = float(recent_data['low'].min())
        resistance = float(recent_data['high'].max())
        
        # Affiner avec les niveaux qui ont été touchés plusieurs fois
        price_levels = pd.concat([recent_data['high'], recent_data['low']])
        price_counts = price_levels.value_counts()
        
        # Si on trouve des niveaux récurrents, les utiliser
        if len(price_counts) > 0:
            # Niveaux les plus touchés
            important_levels = price_counts.head(10).index.tolist()
            current_price = float(df['close'].iloc[-1])
            
            # Support = niveau important le plus proche en dessous
            supports = [p for p in important_levels if p < current_price]
            if supports:
                support = max(supports)
            
            # Résistance = niveau important le plus proche au-dessus
            resistances = [p for p in important_levels if p > current_price]
            if resistances:
                resistance = min(resistances)
        
        return support, resistance
    
    def get_market_conditions(self, symbol: str) -> Dict:
        """
        Analyse les conditions de marché pour un symbole
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Dict avec l'analyse du marché
        """
        indicators = self.calculate_indicators(symbol)
        
        if not indicators:
            return {'status': 'NO_DATA'}
        
        conditions = {
            'symbol': symbol,
            'price': indicators['current_price'],
            'timestamp': datetime.now()
        }
        
        # Tendance
        if indicators['sma_20'] > indicators['sma_50']:
            conditions['trend'] = 'BULLISH'
        else:
            conditions['trend'] = 'BEARISH'
        
        # Force de la tendance
        trend_strength = abs(indicators['sma_20'] - indicators['sma_50']) / indicators['sma_50'] * 100
        if trend_strength > 5:
            conditions['trend_strength'] = 'STRONG'
        elif trend_strength > 2:
            conditions['trend_strength'] = 'MODERATE'
        else:
            conditions['trend_strength'] = 'WEAK'
        
        # Momentum
        if indicators['momentum_10'] > 5:
            conditions['momentum'] = 'STRONG_UP'
        elif indicators['momentum_10'] > 0:
            conditions['momentum'] = 'UP'
        elif indicators['momentum_10'] > -5:
            conditions['momentum'] = 'DOWN'
        else:
            conditions['momentum'] = 'STRONG_DOWN'
        
        # RSI
        if indicators['rsi'] > 70:
            conditions['rsi_status'] = 'OVERBOUGHT'
        elif indicators['rsi'] < 30:
            conditions['rsi_status'] = 'OVERSOLD'
        else:
            conditions['rsi_status'] = 'NEUTRAL'
        
        # Position dans les bandes de Bollinger
        bb_position = (indicators['current_price'] - indicators['bb_lower']) / \
                     (indicators['bb_upper'] - indicators['bb_lower'])
        conditions['bb_position'] = float(bb_position)
        
        # Volatilité
        volatility = indicators['atr'] / indicators['current_price'] * 100
        if volatility > 5:
            conditions['volatility'] = 'HIGH'
        elif volatility > 2:
            conditions['volatility'] = 'MEDIUM'
        else:
            conditions['volatility'] = 'LOW'
        
        # Volume
        if indicators['volume_ratio'] > 2:
            conditions['volume_status'] = 'VERY_HIGH'
        elif indicators['volume_ratio'] > 1.5:
            conditions['volume_status'] = 'HIGH'
        elif indicators['volume_ratio'] < 0.5:
            conditions['volume_status'] = 'LOW'
        else:
            conditions['volume_status'] = 'NORMAL'
        
        return conditions
    
    async def update_market_snapshot(self):
        """Met à jour le snapshot global du marché"""
        try:
            # Récupérer les données BTC
            btc_ticker = await self.get_ticker('BTC/USDT')
            
            if not btc_ticker:
                return
            
            # Récupérer les top gainers/losers
            # Pour l'instant, on simule avec des valeurs
            # TODO: Implémenter la vraie logique
            
            self.market_snapshot = MarketSnapshot(
                timestamp=datetime.now(),
                btc_price=btc_ticker.get('last', 0),
                btc_change_24h=btc_ticker.get('percentage', 0),
                total_market_cap=0,  # TODO: Récupérer depuis CoinGecko
                fear_greed_index=50,  # TODO: Récupérer depuis API
                dominance_btc=0,  # TODO: Calculer
                volume_24h=btc_ticker.get('quoteVolume', 0),
                top_gainers=[],  # TODO: Scanner le marché
                top_losers=[]   # TODO: Scanner le marché
            )
            
            self.last_snapshot_time = datetime.now()
            
        except Exception as e:
            log_error(f"Erreur update market snapshot: {str(e)}")
    
    def get_multi_timeframe_analysis(self, symbol: str) -> Dict:
        """
        Analyse multi-timeframe d'un symbole
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Dict avec l'analyse sur plusieurs timeframes
        """
        analysis = {}
        
        for timeframe in ['15m', '1h', '4h']:
            conditions = self.get_market_conditions(symbol)
            indicators = self.calculate_indicators(symbol, timeframe)
            
            analysis[timeframe] = {
                'trend': conditions.get('trend', 'UNKNOWN'),
                'momentum': conditions.get('momentum', 'UNKNOWN'),
                'rsi': indicators.get('rsi', 50),
                'volume_ratio': indicators.get('volume_ratio', 1)
            }
        
        # Score global basé sur l'alignement des timeframes
        aligned_bullish = all(
            analysis[tf]['trend'] == 'BULLISH' 
            for tf in ['15m', '1h', '4h']
        )
        aligned_bearish = all(
            analysis[tf]['trend'] == 'BEARISH' 
            for tf in ['15m', '1h', '4h']
        )
        
        if aligned_bullish:
            analysis['alignment'] = 'STRONG_BULLISH'
        elif aligned_bearish:
            analysis['alignment'] = 'STRONG_BEARISH'
        else:
            analysis['alignment'] = 'MIXED'
        
        return analysis
    
    def cleanup_old_data(self):
        """Nettoie les données trop anciennes du cache"""
        for symbol in list(self.data.keys()):
            for timeframe in list(self.data[symbol].keys()):
                df = self.data[symbol][timeframe]
                
                if len(df) > self.cache_size:
                    # Garder seulement les N dernières bougies
                    self.data[symbol][timeframe] = df.iloc[-self.cache_size:]
        
        log_debug("Nettoyage du cache de données terminé")