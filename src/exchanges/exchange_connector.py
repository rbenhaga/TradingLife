"""
Module de gestion des connexions aux exchanges
"""

import ccxt
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from src.core.logger import log_info, log_error, log_debug

class ExchangeConnector:
    """Gestionnaire de connexion aux exchanges"""
    
    def __init__(self, exchange_id: str = 'binance', testnet: bool = True):
        """
        Initialise le connecteur d'exchange
        
        Args:
            exchange_id: Identifiant de l'exchange (par défaut: binance)
            testnet: Utiliser le testnet (par défaut: True)
        """
        self.exchange_id = exchange_id
        self.testnet = testnet
        self.exchange = None
        self.connected = False
        
        log_info(f"Initialisation du connecteur {exchange_id} (testnet: {testnet})")
    
    async def connect(self, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> bool:
        """
        Établit la connexion avec l'exchange
        
        Args:
            api_key: Clé API (optionnel pour le testnet)
            api_secret: Secret API (optionnel pour le testnet)
            
        Returns:
            True si la connexion est réussie
        """
        try:
            # Créer l'instance de l'exchange
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'testnet': self.testnet
                }
            })
            
            # Tester la connexion
            await self.exchange.load_markets()
            self.connected = True
            
            log_info(f"Connexion réussie à {self.exchange_id}")
            return True
            
        except Exception as e:
            log_error(f"Erreur de connexion à {self.exchange_id}: {str(e)}")
            self.connected = False
            return False
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Récupère le ticker pour une paire
        
        Args:
            symbol: Symbole de la paire (ex: BTC/USDT)
            
        Returns:
            Dictionnaire contenant les données du ticker ou None en cas d'erreur
        """
        if not self.connected:
            log_error("Non connecté à l'exchange")
            return None
            
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            log_error(f"Erreur lors de la récupération du ticker {symbol}: {str(e)}")
            return None
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '15m', 
                        limit: int = 100) -> Optional[List]:
        """
        Récupère les données OHLCV
        
        Args:
            symbol: Symbole de la paire
            timeframe: Période des bougies
            limit: Nombre de bougies à récupérer
            
        Returns:
            Liste des bougies ou None en cas d'erreur
        """
        if not self.connected:
            log_error("Non connecté à l'exchange")
            return None
            
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            log_error(f"Erreur lors de la récupération des OHLCV {symbol}: {str(e)}")
            return None
    
    async def get_balance(self, currency: str = 'USDT') -> Optional[float]:
        """
        Récupère le solde d'une devise
        
        Args:
            currency: Code de la devise (par défaut: USDT)
            
        Returns:
            Solde disponible ou None en cas d'erreur
        """
        if not self.connected:
            log_error("Non connecté à l'exchange")
            return None
            
        try:
            balance = await self.exchange.fetch_balance()
            return float(balance.get(currency, {}).get('free', 0))
        except Exception as e:
            log_error(f"Erreur lors de la récupération du solde {currency}: {str(e)}")
            return None
    
    async def create_order(self, symbol: str, order_type: str, side: str, 
                          amount: float, price: Optional[float] = None) -> Optional[Dict]:
        """
        Crée un ordre
        
        Args:
            symbol: Symbole de la paire
            order_type: Type d'ordre (market, limit)
            side: Côté de l'ordre (buy, sell)
            amount: Quantité
            price: Prix (requis pour les ordres limit)
            
        Returns:
            Dictionnaire contenant les détails de l'ordre ou None en cas d'erreur
        """
        if not self.connected:
            log_error("Non connecté à l'exchange")
            return None
            
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price
            )
            log_info(f"Ordre créé: {symbol} {side} {amount} @ {price if price else 'market'}")
            return order
        except Exception as e:
            log_error(f"Erreur lors de la création de l'ordre {symbol}: {str(e)}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Annule un ordre
        
        Args:
            order_id: ID de l'ordre
            symbol: Symbole de la paire
            
        Returns:
            True si l'ordre a été annulé
        """
        if not self.connected:
            log_error("Non connecté à l'exchange")
            return False
            
        try:
            await self.exchange.cancel_order(order_id, symbol)
            log_info(f"Ordre annulé: {order_id} ({symbol})")
            return True
        except Exception as e:
            log_error(f"Erreur lors de l'annulation de l'ordre {order_id}: {str(e)}")
            return False
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Dict]:
        """
        Récupère les détails d'un ordre
        
        Args:
            order_id: ID de l'ordre
            symbol: Symbole de la paire
            
        Returns:
            Dictionnaire contenant les détails de l'ordre ou None en cas d'erreur
        """
        if not self.connected:
            log_error("Non connecté à l'exchange")
            return None
            
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            log_error(f"Erreur lors de la récupération de l'ordre {order_id}: {str(e)}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les ordres ouverts
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Liste des ordres ouverts
        """
        if not self.connected:
            log_error("Non connecté à l'exchange")
            return []
            
        try:
            orders = await self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            log_error(f"Erreur lors de la récupération des ordres ouverts: {str(e)}")
            return []
    
    async def close(self):
        """Ferme la connexion à l'exchange"""
        if self.connected:
            self.exchange = None
            self.connected = False
            log_info(f"Déconnexion de {self.exchange_id}") 