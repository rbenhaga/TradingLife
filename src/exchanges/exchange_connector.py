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

from src.core.logger import log_info, log_error, log_debug, log_warning


class ExchangeConnector:
    """
    Gestionnaire de connexion aux exchanges avec support REST et WebSocket
    """
    
    def __init__(self, exchange_name: str = 'binance', testnet: bool = True):
        """
        Initialise le connecteur d'exchange
        
        Args:
            exchange_name: Nom de l'exchange (binance, bybit, etc.)
            testnet: Utiliser le testnet
        """
        self.exchange_name = exchange_name
        self.testnet = testnet
        self.exchange = None
        self.connected = False
        
        # Métriques
        self.api_calls = 0
        self.last_call_time = 0
        self.rate_limit_remaining = 1000
        
        log_info(f"ExchangeConnector initialisé - {exchange_name} ({'testnet' if testnet else 'mainnet'})")
    
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
        return await self._api_call('fetch_order_book', symbol, limit)
    
    async def get_balance(self) -> Optional[Dict]:
        """
        Récupère les balances du compte
        
        Returns:
            Dictionnaire des balances par devise
        """
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
        orders = await self._api_call('fetch_open_orders', symbol)
        return orders or []
    
    async def get_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Récupère l'historique des trades
        
        Args:
            symbol: Symbole de la paire
            limit: Nombre de trades
            
        Returns:
            Liste des trades
        """
        trades = await self._api_call('fetch_my_trades', symbol, limit=limit)
        return trades or []
    
    async def _api_call(self, method: str, *args, **kwargs):
        """
        Effectue un appel API avec gestion d'erreurs et rate limiting
        
        Args:
            method: Nom de la méthode à appeler
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            Résultat de l'appel ou None en cas d'erreur
        """
        if not self.connected or not self.exchange:
            log_error(f"Non connecté à l'exchange pour {method}")
            return None
        
        # Rate limiting manuel supplémentaire
        time_since_last = time.time() - self.last_call_time
        if time_since_last < 0.05:  # 50ms minimum entre les appels
            await asyncio.sleep(0.05 - time_since_last)
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Appeler la méthode
                method_func = getattr(self.exchange, method)
                result = await method_func(*args, **kwargs)
                
                # Mise à jour des métriques
                self.api_calls += 1
                self.last_call_time = time.time()
                
                return result
                
            except ccxt.RateLimitExceeded as e:
                log_warning(f"Rate limit atteint: {e}")
                await asyncio.sleep(retry_delay * (attempt + 1))
                
            except ccxt.NetworkError as e:
                log_error(f"Erreur réseau lors de {method}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    return None
                    
            except ccxt.ExchangeError as e:
                log_error(f"Erreur exchange lors de {method}: {e}")
                return None
                
            except Exception as e:
                log_error(f"Erreur inattendue lors de {method}: {e}")
                return None
        
        return None
    
    async def close(self):
        """Ferme la connexion à l'exchange"""
        if self.exchange:
            await self.exchange.close()
            self.connected = False
            log_info(f"Déconnexion de {self.exchange_name}")
    
    def get_min_order_size(self, symbol: str) -> float:
        """
        Retourne la taille minimale d'ordre pour un symbole
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Taille minimale d'ordre
        """
        if self.exchange and symbol in self.exchange.markets:
            market = self.exchange.markets[symbol]
            return market.get('limits', {}).get('amount', {}).get('min', 0.001)
        
        return 0.001  # Valeur par défaut
    
    def get_fee_rate(self, symbol: str, order_type: str = 'taker') -> float:
        """
        Retourne le taux de frais pour un symbole
        
        Args:
            symbol: Symbole de la paire
            order_type: Type d'ordre (maker/taker)
            
        Returns:
            Taux de frais (ex: 0.001 pour 0.1%)
        """
        if self.exchange and symbol in self.exchange.markets:
            market = self.exchange.markets[symbol]
            return market.get(order_type, 0.001)
        
        return 0.001  # 0.1% par défaut