"""
Module de scan de watchlist pour le bot de trading
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.exchanges.exchange_connector import ExchangeConnector
from src.core.logger import log_info, log_error, log_debug
from src.core.market_data import MarketData

logger = logging.getLogger(__name__)

class WatchlistScanner:
    """Scanner de watchlist pour identifier les opportunités de trading"""
    
    def __init__(self, exchange_connector: ExchangeConnector, min_volume_usdt: float = 1_000_000, top_n: int = 10):
        """
        Initialise le scanner de watchlist
        
        Args:
            exchange_connector: Connecteur d'exchange
            min_volume_usdt: Volume minimum en USDT
            top_n: Nombre de paires à sélectionner
        """
        self.exchange = exchange_connector
        self.min_volume_usdt = min_volume_usdt
        self.top_n = top_n
        self.watchlist: List[str] = []
        self.last_update: Optional[datetime] = None
        self.update_interval = timedelta(minutes=15)
        
        log_info(f"WatchlistScanner initialisé - Volume min: {min_volume_usdt} USDT, Top {top_n}")
    
    async def update_watchlist(self) -> List[str]:
        """
        Met à jour la watchlist en fonction des volumes et de la volatilité
        
        Returns:
            Liste des paires à surveiller
        """
        try:
            if (self.last_update and 
                datetime.now() - self.last_update < self.update_interval):
                return self.watchlist
            
            # Récupérer toutes les paires disponibles
            tickers = await self.exchange.get_tickers()
            
            # Filtrer et trier les paires
            valid_pairs = []
            for symbol, ticker in tickers.items():
                if not symbol.endswith('USDT'):
                    continue
                    
                volume_usdt = float(ticker['quoteVolume'])
                if volume_usdt < self.min_volume_usdt:
                    continue
                    
                valid_pairs.append({
                    'symbol': symbol,
                    'volume': volume_usdt,
                    'volatility': abs(float(ticker['priceChangePercent']))
                })
            
            # Trier par volume et volatilité
            valid_pairs.sort(
                key=lambda x: (x['volume'], x['volatility']),
                reverse=True
            )
            
            # Sélectionner le top N
            self.watchlist = [p['symbol'] for p in valid_pairs[:self.top_n]]
            self.last_update = datetime.now()
            
            log_info(f"Watchlist mise à jour - {len(self.watchlist)} paires sélectionnées")
            return self.watchlist
            
        except Exception as e:
            log_error(f"Erreur lors de la mise à jour de la watchlist: {str(e)}")
            return []
    
    async def get_watchlist(self) -> List[str]:
        """
        Retourne la watchlist actuelle
        
        Returns:
            Liste des paires à surveiller
        """
        if not self.watchlist or not self.last_update:
            return await self.update_watchlist()
        return self.watchlist
    
    async def is_valid_pair(self, symbol: str) -> bool:
        """
        Vérifie si une paire est valide pour le trading
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            True si la paire est valide
        """
        try:
            ticker = await self.exchange.get_ticker(symbol)
            if not ticker:
                return False
                
            volume_usdt = float(ticker['quoteVolume'])
            return volume_usdt >= self.min_volume_usdt
            
        except Exception as e:
            log_error(f"Erreur lors de la vérification de la paire {symbol}: {str(e)}")
            return False

    def get_trading_params(self, symbol: str) -> dict:
        """
        Retourne les paramètres de trading pour une paire (valeurs par défaut)
        """
        return {
            'timeframe': '15m',
            'position_size': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.04
        }

    def get_pair_metrics(self, symbol: str) -> dict:
        """
        Retourne des métriques fictives pour une paire (valeurs par défaut)
        """
        return {
            'volatility': 1.0
        }

# Ajouter dans watchlist_scanner.py
async def calculate_volatility_metrics(self, symbol: str) -> Dict:
    """Calcule les métriques de volatilité avancées"""
    try:
        # Récupérer les données 15m
        ohlcv = await self.exchange.get_ohlcv(symbol, '15m', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # ATR pour la volatilité
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        atr_pct = (atr / df['close'].iloc[-1]) * 100
        
        # Volume profile
        volume_ma = df['volume'].rolling(20).mean()
        volume_spike = df['volume'].iloc[-1] / volume_ma.iloc[-1]
        
        # Momentum
        momentum = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
        
        return {
            'atr_pct': atr_pct,
            'volume_spike': volume_spike,
            'momentum_20': momentum,
            'volatility_score': atr_pct * volume_spike  # Score composite
        }
    except Exception as e:
        log_error(f"Erreur calcul volatilité {symbol}: {e}")
        return {}