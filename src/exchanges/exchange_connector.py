"""
Module de gestion des connexions aux exchanges
Version améliorée avec support WebSocket intégré
"""

import ccxt
import ccxt.async_support as ccxt_async
import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime
import time

from ..core.logger import log_info, log_error, log_debug, log_warning


class ExchangeConnector:
    """
    Gestionnaire de connexion aux exchanges avec support REST et WebSocket
    """
    
    def __init__(self, exchange_name: str = 'binance', testnet: bool = True, skip_connection: bool = False):
        """
        Initialise le connecteur d'exchange
        
        Args:
            exchange_name: Nom de l'exchange (binance, bybit, etc.)
            testnet: Utiliser le testnet
            skip_connection: Ne pas se connecter à l'exchange (mode simulation)
        """
        self.exchange_name = exchange_name
        self.testnet = testnet
        self.skip_connection = skip_connection
        self.exchange = None
        self.connected = skip_connection  # Si skip_connection, on est considéré comme connecté
        
        # Métriques
        self.api_calls = 0
        self.last_call_time = 0
        self.rate_limit_remaining = 1000
        
        log_info(f"ExchangeConnector initialisé - {exchange_name} ({'testnet' if testnet else 'mainnet'})")
        if skip_connection:
            log_info("Mode simulation activé - Pas de connexion à l'exchange")
    
    async def connect(self, api_key: Optional[str] = None, 
                     api_secret: Optional[str] = None) -> bool:
        """
        Établit la connexion avec l'exchange
        
        Args:
            api_key: Clé API
            api_secret: Secret API
            
        Returns:
            True si la connexion est réussie
        """
        if self.skip_connection:
            log_info("Mode simulation - Pas de connexion à l'exchange")
            return True
            
        try:
            # Créer l'instance async de l'exchange
            exchange_class = getattr(ccxt_async, self.exchange_name)
            
            config = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'rateLimit': 50,  # ms entre les requêtes
                'options': {
                    'defaultType': 'spot',
                }
            }
            
            # Configuration spécifique au testnet
            if self.testnet:
                if self.exchange_name == 'binance':
                    config['urls'] = {
                        'api': {
                            'public': 'https://testnet.binance.vision/api/v3',
                            'private': 'https://testnet.binance.vision/api/v3',
                        }
                    }
                elif self.exchange_name == 'bybit':
                    config['options']['testnet'] = True
            
            self.exchange = exchange_class(config)
            
            # Charger les marchés
            await self.exchange.load_markets()
            self.connected = True
            
            # Test de connexion
            balance = await self.exchange.fetch_balance()
            log_info(f"✅ Connecté à {self.exchange_name} - Balance USDT: {balance.get('USDT', {}).get('free', 0):.2f}")
            
            return True
            
        except Exception as e:
            log_error(f"Erreur de connexion à {self.exchange_name}: {str(e)}")
            self.connected = False
            return False
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Récupère le ticker pour une paire
        
        Args:
            symbol: Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            Dictionnaire contenant les données du ticker
        """
        if self.skip_connection:
            return {
                'symbol': symbol,
                'last': 50000.0,  # Prix simulé
                'bid': 49999.0,
                'ask': 50001.0,
                'volume': 1000.0
            }
        return await self._api_call('fetch_ticker', symbol)
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '15m', 
                       limit: int = 100) -> Optional[List]:
        """
        Récupère les données OHLCV
        
        Args:
            symbol: Symbole de la paire
            timeframe: Période des bougies
            limit: Nombre de bougies à récupérer
            
        Returns:
            Liste des bougies OHLCV
        """
        if self.skip_connection:
            # Générer des données OHLCV simulées
            now = int(time.time() * 1000)
            data = []
            for i in range(limit):
                timestamp = now - (limit - i) * 15 * 60 * 1000  # 15 minutes
                data.append([
                    timestamp,
                    50000.0 + i * 10,  # Open
                    50100.0 + i * 10,  # High
                    49900.0 + i * 10,  # Low
                    50050.0 + i * 10,  # Close
                    1000.0  # Volume
                ])
            return data
        return await self._api_call('fetch_ohlcv', symbol, timeframe, limit=limit)
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """
        Récupère le carnet d'ordres
        
        Args:
            symbol: Symbole de la paire
            limit: Profondeur du carnet
            
        Returns:
            Carnet d'ordres
        """
        if self.skip_connection:
            return {
                'bids': [[50000.0 - i, 1.0] for i in range(limit)],
                'asks': [[50000.0 + i, 1.0] for i in range(limit)]
            }
        return await self._api_call('fetch_order_book', symbol, limit)
    
    async def get_balance(self) -> Optional[Dict]:
        """
        Récupère les balances du compte
        
        Returns:
            Dictionnaire des balances par devise
        """
        if self.skip_connection:
            return {
                'USDT': {'free': 10000.0, 'used': 0.0, 'total': 10000.0},
                'BTC': {'free': 1.0, 'used': 0.0, 'total': 1.0}
            }
        return await self._api_call('fetch_balance')
    
    async def create_order(self, symbol: str, order_type: str, side: str, 
                          amount: float, price: Optional[float] = None,
                          params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Crée un ordre
        
        Args:
            symbol: Symbole de la paire
            order_type: Type d'ordre (market, limit)
            side: Côté de l'ordre (buy, sell)
            amount: Quantité
            price: Prix (requis pour les ordres limit)
            params: Paramètres additionnels
            
        Returns:
            Détails de l'ordre créé
        """
        if self.skip_connection:
            order = {
                'id': f"sim_{int(time.time() * 1000)}",
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price or 50000.0,
                'status': 'closed',
                'filled': amount,
                'remaining': 0.0,
                'cost': amount * (price or 50000.0),
                'timestamp': int(time.time() * 1000)
            }
            log_info(
                f"📝 Ordre simulé créé: {symbol} {side} {amount} @ "
                f"{price if price else 'market'} | ID: {order['id']}"
            )
            return order
            
        order = await self._api_call(
            'create_order',
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=price,
            params=params or {}
        )
        
        if order:
            log_info(
                f"📝 Ordre créé: {symbol} {side} {amount} @ "
                f"{price if price else 'market'} | ID: {order.get('id')}"
            )
        
        return order
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Annule un ordre
        
        Args:
            order_id: ID de l'ordre
            symbol: Symbole de la paire
            
        Returns:
            True si l'ordre a été annulé
        """
        if self.skip_connection:
            log_info(f"❌ Ordre simulé annulé: {order_id} ({symbol})")
            return True
            
        result = await self._api_call('cancel_order', order_id, symbol)
        
        if result:
            log_info(f"❌ Ordre annulé: {order_id} ({symbol})")
            return True
        
        return False
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les ordres ouverts
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Liste des ordres ouverts
        """
        if self.skip_connection:
            return []
        return await self._api_call('fetch_open_orders', symbol)
    
    async def get_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Récupère l'historique des trades
        
        Args:
            symbol: Symbole de la paire
            limit: Nombre de trades à récupérer
            
        Returns:
            Liste des trades
        """
        if self.skip_connection:
            return []
        return await self._api_call('fetch_trades', symbol, limit=limit)
    
    async def _api_call(self, method: str, *args, **kwargs):
        """
        Effectue un appel API avec gestion des erreurs et rate limiting
        
        Args:
            method: Nom de la méthode à appeler
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            Résultat de l'appel API
        """
        if not self.connected:
            log_error(f"Non connecté à {self.exchange_name}")
            return None
            
        if not self.exchange:
            log_error(f"Exchange non initialisé")
            return None
            
        try:
            # Vérifier le rate limit
            now = time.time()
            if now - self.last_call_time < 0.05:  # 50ms minimum entre les appels
                await asyncio.sleep(0.05)
            
            # Appel API
            method = getattr(self.exchange, method)
            result = await method(*args, **kwargs)
            
            # Mise à jour des métriques
            self.api_calls += 1
            self.last_call_time = time.time()
            
            return result
            
        except ccxt.NetworkError as e:
            log_error(f"Erreur réseau: {str(e)}")
            return None
        except ccxt.ExchangeError as e:
            log_error(f"Erreur exchange: {str(e)}")
            return None
        except Exception as e:
            log_error(f"Erreur inattendue: {str(e)}")
            return None
    
    async def close(self):
        """Ferme la connexion à l'exchange"""
        if self.exchange and not self.skip_connection:
            await self.exchange.close()
            self.connected = False
            log_info(f"Déconnexion de {self.exchange_name}")
    
    def get_min_order_size(self, symbol: str) -> float:
        """
        Récupère la taille minimale d'ordre pour une paire
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Taille minimale d'ordre
        """
        if self.skip_connection:
            return 0.001  # 0.001 BTC par défaut
            
        if not self.exchange or not self.exchange.markets:
            return 0.001
            
        market = self.exchange.markets.get(symbol)
        if not market:
            return 0.001
            
        return float(market.get('limits', {}).get('amount', {}).get('min', 0.001))
    
    def get_fee_rate(self, symbol: str, order_type: str = 'taker') -> float:
        """
        Récupère le taux de frais pour une paire
        
        Args:
            symbol: Symbole de la paire
            order_type: Type d'ordre (maker/taker)
            
        Returns:
            Taux de frais (ex: 0.001 pour 0.1%)
        """
        if self.skip_connection:
            return 0.001  # 0.1% par défaut
            
        if not self.exchange or not self.exchange.markets:
            return 0.001
            
        market = self.exchange.markets.get(symbol)
        if not market:
            return 0.001
            
        fees = market.get('taker' if order_type == 'taker' else 'maker', 0.001)
        return float(fees)

    async def get_tickers(self) -> Dict:
        """
        Récupère les tickers pour toutes les paires
        
        Returns:
            Dict des tickers avec leurs données
        """
        try:
            if self.skip_connection:
                # Retourner des données simulées
                return {
                    'BTCUSDT': {
                        'symbol': 'BTCUSDT',
                        'price': 50000.0,
                        'quoteVolume': 1000000000.0,
                        'priceChangePercent': 2.5
                    },
                    'ETHUSDT': {
                        'symbol': 'ETHUSDT',
                        'price': 3000.0,
                        'quoteVolume': 500000000.0,
                        'priceChangePercent': 1.8
                    }
                }
            
            # Récupérer les tickers de l'exchange
            tickers = await self.exchange.fetch_tickers()
            
            # Formater les données
            formatted_tickers = {}
            for symbol, ticker in tickers.items():
                if not symbol.endswith('USDT'):
                    continue
                    
                formatted_tickers[symbol] = {
                    'symbol': symbol,
                    'price': float(ticker['last']),
                    'quoteVolume': float(ticker['quoteVolume']),
                    'priceChangePercent': float(ticker['percentage'])
                }
            
            return formatted_tickers
            
        except Exception as e:
            log_error(f"Erreur lors de la récupération des tickers: {str(e)}")
            return {}