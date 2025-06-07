#!/usr/bin/env python3
"""
Module d'interaction avec l'API Binance
Optimisé pour le testnet SPOT uniquement
"""

import ccxt
from typing import Dict, List, Optional
import asyncio
import aiohttp
from datetime import datetime
import os
from dotenv import load_dotenv

# Charger les variables d'environnement une seule fois
load_dotenv()

class BinanceTestnetConnector:
    """Connecteur optimisé pour Binance Testnet SPOT"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.base_url = "https://testnet.binance.vision/api/v3"
        self.exchange = None
        self._session = None
        
    def create_exchange(self) -> ccxt.binance:
        """Crée une instance de l'exchange Binance en mode testnet SPOT uniquement"""
        if self.exchange:
            return self.exchange
            
        # Configuration spéciale pour le testnet Binance
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'rateLimit': 50,  # Plus agressif pour le testnet
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'createMarketBuyOrderRequiresPrice': False,
                'fetchMarkets': {'type': 'spot'},
                'recvWindow': 60000  # 60 secondes pour éviter les erreurs de timing
            }
        })
        
        # Configuration sandbox SPOT uniquement
        self.exchange.set_sandbox_mode(True)
        self.exchange.urls['api'] = {
            'public': self.base_url,
            'private': self.base_url,
            'v1': 'https://testnet.binance.vision/api/v1'
        }
        
        # Important: supprimer toute référence aux futures
        if 'fapiPublic' in self.exchange.urls:
            del self.exchange.urls['fapiPublic']
        if 'fapiPrivate' in self.exchange.urls:
            del self.exchange.urls['fapiPrivate']
            
        return self.exchange
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Retourne une session aiohttp réutilisable"""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close_session(self):
        """Ferme la session aiohttp"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def fetch_ticker_async(self, symbol: str) -> Optional[Dict]:
        """Récupère le ticker d'une paire de manière asynchrone"""
        session = await self.get_session()
        formatted_symbol = symbol.replace('/', '')
        
        try:
            url = f"{self.base_url}/ticker/24hr?symbol={formatted_symbol}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'symbol': symbol,
                        'last': float(data['lastPrice']),
                        'high': float(data['highPrice']),
                        'low': float(data['lowPrice']),
                        'volume': float(data['volume']),
                        'quoteVolume': float(data['quoteVolume']),
                        'percentage': float(data['priceChangePercent']),
                        'bid': float(data['bidPrice']),
                        'ask': float(data['askPrice']),
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            print(f"Erreur fetch ticker {symbol}: {e}")
        return None
    
    async def fetch_multiple_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
        """Récupère plusieurs tickers en parallèle"""
        tasks = [self.fetch_ticker_async(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return {
            symbol: ticker 
            for symbol, ticker in zip(symbols, results) 
            if ticker is not None
        }
    
    def get_testnet_balance(self) -> Dict:
        """Récupère la balance du compte testnet"""
        if not self.exchange:
            self.create_exchange()
            
        try:
            response = self.exchange.privateGetAccount()
            balances = response.get('balances', [])
            
            # Filtrer et formater les balances
            balance_dict = {}
            for balance in balances:
                free = float(balance.get('free', 0))
                locked = float(balance.get('locked', 0))
                total = free + locked
                
                if total > 0:
                    balance_dict[balance['asset']] = {
                        'free': free,
                        'locked': locked,
                        'total': total,
                        'percentage': (total / 10000 * 100) if balance['asset'] == 'USDT' else 0
                    }
            
            return balance_dict
        except Exception as e:
            raise Exception(f"Erreur balance: {str(e)}")
    
    def get_testnet_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """Récupère les données OHLCV du testnet"""
        if not self.exchange:
            self.create_exchange()
            
        try:
            # Format Binance
            formatted_symbol = symbol.replace('/', '')
            
            response = self.exchange.publicGetKlines({
                'symbol': formatted_symbol,
                'interval': interval,
                'limit': limit
            })
            
            # Convertir avec calculs supplémentaires
            klines = []
            for i, candle in enumerate(response):
                kline = {
                    'timestamp': int(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]),
                    'trades': int(candle[8]) if len(candle) > 8 else 0
                }
                
                # Calculer des métriques supplémentaires
                kline['range'] = kline['high'] - kline['low']
                kline['range_pct'] = (kline['range'] / kline['close'] * 100) if kline['close'] > 0 else 0
                kline['body'] = abs(kline['close'] - kline['open'])
                kline['body_pct'] = (kline['body'] / kline['close'] * 100) if kline['close'] > 0 else 0
                
                # Direction de la bougie
                kline['direction'] = 'up' if kline['close'] > kline['open'] else 'down'
                
                klines.append(kline)
            
            return klines
        except Exception as e:
            raise Exception(f"Erreur klines: {str(e)}")
    
    async def place_test_order(self, symbol: str, side: str, quantity: float, price: float = None) -> Dict:
        """Place un ordre de test (ne sera pas exécuté réellement)"""
        if not self.exchange:
            self.create_exchange()
            
        try:
            formatted_symbol = symbol.replace('/', '')
            
            params = {
                'symbol': formatted_symbol,
                'side': side.upper(),
                'type': 'LIMIT' if price else 'MARKET',
                'quantity': quantity,
                'recvWindow': 60000
            }
            
            if price:
                params['price'] = price
                params['timeInForce'] = 'GTC'
            
            # Utiliser l'endpoint de test
            response = self.exchange.privatePostOrderTest(params)
            
            return {
                'status': 'success',
                'test': True,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }

# Instance globale pour réutilisation
_connector = None

def get_connector() -> BinanceTestnetConnector:
    """Retourne l'instance unique du connecteur"""
    global _connector
    if not _connector:
        _connector = BinanceTestnetConnector()
    return _connector

# Fonctions de compatibilité avec l'ancien code
def create_binance_testnet_exchange() -> ccxt.binance:
    """Fonction de compatibilité"""
    return get_connector().create_exchange()

def get_testnet_balance(exchange: ccxt.binance = None) -> Dict:
    """Fonction de compatibilité"""
    return get_connector().get_testnet_balance()

def get_testnet_klines(exchange: ccxt.binance, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
    """Fonction de compatibilité"""
    return get_connector().get_testnet_klines(symbol, interval, limit)