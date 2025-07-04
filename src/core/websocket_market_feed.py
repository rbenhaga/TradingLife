"""
Module WebSocket pour données de marché temps réel haute performance
Version corrigée pour Binance Testnet
"""

import asyncio
import json
import time
from typing import Dict, List, Callable, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import websockets
import aiohttp
from enum import Enum

from .logger import log_info, log_error, log_warning, log_debug
from .latency_monitor import LatencyMonitor

class DataType(Enum):
    """Types de données supportés"""
    TICKER = "ticker"
    TRADES = "trades"
    ORDERBOOK = "orderbook"
    KLINES = "klines"
    DEPTH = "depth"

@dataclass
class MarketUpdate:
    """Mise à jour du marché"""
    symbol: str
    data_type: DataType
    data: Dict
    timestamp: float
    latency_ms: float
    exchange: str = "binance"

@dataclass
class OrderBookSnapshot:
    """Snapshot du carnet d'ordres"""
    symbol: str
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]
    timestamp: float
    update_id: int
    
class WebSocketMarketFeed:
    """
    Feed de marché WebSocket haute performance
    """
    
    # URLs WebSocket Binance
    BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
    BINANCE_WS_STREAM = "wss://stream.binance.com:9443/stream"
    # Binance Testnet utilise les mêmes endpoints que la production pour le WebSocket
    # Mais avec des données différentes selon les clés API utilisées
    
    def __init__(self, exchange: str = "binance", testnet: bool = False,
                 max_reconnect_attempts: int = 5):
        """
        Initialise le feed WebSocket
        
        Args:
            exchange: Exchange à utiliser
            testnet: Utiliser le testnet
            max_reconnect_attempts: Tentatives max de reconnexion
        """
        self.exchange = exchange
        self.testnet = testnet
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # WebSocket URL
        self.ws_url = self.BINANCE_WS_BASE
        
        # État de connexion
        self.websocket = None
        self.connected = False
        self.reconnect_count = 0
        self.last_heartbeat = time.time()
        self.heartbeat_timeout = 30  # secondes
        
        # Gestion des limites de taux
        self.last_subscription_time = 0
        self.subscription_delay = 1.0
        self.max_subscriptions_per_second = 5
        self.subscription_count = 0
        self.subscription_reset_time = time.time()
        self.pending_subscriptions = []
        
        # Subscriptions
        self.subscriptions: Dict[str, Set[DataType]] = defaultdict(set)
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Métriques de performance
        self.latency_buffer = deque(maxlen=1000)
        self.message_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Cache
        self.ticker_cache: Dict[str, Dict] = {}
        self.orderbook_cache: Dict[str, OrderBookSnapshot] = {}
        
        # Flags de contrôle
        self._running = False
        self._reconnect_task = None
        self._heartbeat_task = None
        self._receive_task = None
        self._subscription_task = None

        # Ajout du suivi de latence
        self.latency_monitor = LatencyMonitor()

        log_info(f"WebSocketMarketFeed initialisé - Exchange: {exchange}, Testnet: {testnet}")

    async def connect(self) -> bool:
        """
        Établit la connexion WebSocket avec gestion robuste des erreurs
        
        Returns:
            True si connecté avec succès
        """
        if self._running:
            log_warning("WebSocket déjà en cours d'exécution")
            return True
            
        try:
            self._running = True
            url = self.ws_url
            
            log_info(f"Connexion WebSocket à: {url}")
            
            # Configuration robuste de la connexion
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                    max_size=2**20,  # 1MB max message size
                    max_queue=2**10,  # 1024 messages queue
                    compression=None  # Désactiver la compression pour réduire la latence
                ),
                timeout=10.0
            )
            
            self.connected = True
            self.reconnect_count = 0
            self.last_heartbeat = time.time()

            log_info("✅ WebSocket connecté avec succès")

            # Démarrer les tâches de maintenance
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Re-souscrire aux streams existants
            if self.subscriptions:
                await self._resubscribe_all()
            
            return True
            
        except asyncio.TimeoutError:
            log_error("Timeout lors de la connexion WebSocket")
            self._handle_connection_error("Timeout")
            return False
        except websockets.exceptions.InvalidStatusCode as e:
            log_error(f"Erreur de statut HTTP: {e.status_code}")
            self._handle_connection_error(f"HTTP {e.status_code}")
            return False
        except Exception as e:
            log_error(f"Erreur de connexion WebSocket: {e}")
            self._handle_connection_error(str(e))
            return False

    def _handle_connection_error(self, error_msg: str):
        """Gère les erreurs de connexion"""
        self.connected = False
        self._running = False
        
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_with_backoff(error_msg))

    async def _reconnect_with_backoff(self, error_msg: str):
        """Tente de reconnecter avec backoff exponentiel"""
        if self.reconnect_count >= self.max_reconnect_attempts:
            log_error(f"Nombre maximum de tentatives de reconnexion atteint ({error_msg})")
            return
            
        self.reconnect_count += 1
        wait_time = min(2 ** self.reconnect_count, 60)  # Backoff exponentiel max 60s
        
        log_warning(f"Tentative de reconnexion {self.reconnect_count}/{self.max_reconnect_attempts} dans {wait_time}s")
        await asyncio.sleep(wait_time)
        
        if await self.connect():
            log_info("Reconnexion réussie")
            self.reconnect_count = 0
        else:
            await self._reconnect_with_backoff(error_msg)

    async def disconnect(self):
        """Ferme proprement la connexion WebSocket"""
        self._running = False
        self.connected = False
        
        # Annuler les tâches en cours
        for task in [self._receive_task, self._heartbeat_task, self._reconnect_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Fermer la connexion WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                log_error(f"Erreur lors de la fermeture WebSocket: {e}")
            finally:
                self.websocket = None
        
        log_info("WebSocket déconnecté proprement")

    async def _heartbeat_loop(self):
        """Envoie des pings périodiques et vérifie la connexion"""
        while self._running:
            try:
                await asyncio.sleep(10)  # Ping toutes les 10 secondes
                
                if not self.websocket:
                    raise Exception("WebSocket non initialisé")
                    
                # Vérifier le timeout
                if time.time() - self.last_heartbeat > self.heartbeat_timeout:
                    raise Exception("Timeout heartbeat")
                
                # Envoyer le ping
                pong = await self.websocket.ping()
                await pong
                self.last_heartbeat = time.time()
                
            except Exception as e:
                log_error(f"Erreur heartbeat: {e}")
                self._handle_connection_error(str(e))
                break

    async def _receive_loop(self):
        """Boucle de réception des messages avec gestion robuste des erreurs"""
        while self._running:
            try:
                if not self.websocket:
                    raise Exception("WebSocket non initialisé")
                
                # Recevoir avec timeout
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=30.0
                )
                
                receive_time = time.time()
                self.last_heartbeat = receive_time
                
                # Parser le JSON
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    log_error(f"Erreur de parsing JSON: {e}")
                    continue
                
                # Ignorer les réponses de souscription
                if 'result' in data or 'id' in data:
                    continue
                
                # Traiter le message
                await self._process_message(data, receive_time)
                self.message_count += 1
                
            except asyncio.TimeoutError:
                log_warning("Timeout de réception WebSocket")
                self._handle_connection_error("Receive timeout")
                break
            except websockets.ConnectionClosed:
                log_warning("Connexion WebSocket fermée")
                self._handle_connection_error("Connection closed")
                break
            except Exception as e:
                log_error(f"Erreur dans receive_loop: {e}")
                self.error_count += 1
                await asyncio.sleep(0.1)

    async def _resubscribe_all(self):
        """Re-souscrit à tous les streams après reconnexion"""
        log_info(f"Re-souscription à {len(self.subscriptions)} paires")
        
        # Réinitialiser la file d'attente
        self.pending_subscriptions = []
        
        # Ajouter toutes les souscriptions à la file
        for symbol, data_types in self.subscriptions.items():
            for data_type in data_types:
                self.pending_subscriptions.append((symbol, data_type))
        
        # Démarrer le traitement des souscriptions
        if self.pending_subscriptions:
            self._subscription_task = asyncio.create_task(self._process_subscriptions())
        
        log_info("Re-souscription en cours...")

    async def _process_subscriptions(self):
        """Traite les souscriptions en attente avec gestion des limites"""
        while self.pending_subscriptions:
            current_time = time.time()
            
            # Réinitialiser le compteur si nécessaire
            if current_time - self.subscription_reset_time >= 1.0:
                self.subscription_count = 0
                self.subscription_reset_time = current_time
            
            # Vérifier la limite de taux
            if self.subscription_count >= self.max_subscriptions_per_second:
                wait_time = 1.0 - (current_time - self.subscription_reset_time)
                if wait_time > 0:
                    log_debug(f"Limite de taux atteinte, attente de {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                self.subscription_count = 0
                self.subscription_reset_time = time.time()
            
            # Traiter la prochaine souscription
            symbol, data_type = self.pending_subscriptions.pop(0)
            await self._subscribe_to_stream(symbol, data_type)
            self.subscription_count += 1
            
            # Attendre le délai minimum entre les souscriptions
            await asyncio.sleep(self.subscription_delay)

    async def _subscribe_to_stream(self, symbol: str, data_type: DataType):
        """Souscrit à un stream spécifique"""
        if not self.websocket:
            return
        
        # Construire le nom du stream
        symbol_lower = symbol.replace('/', '').lower()
        
        stream_name = ""
        if data_type == DataType.TICKER:
            stream_name = f"{symbol_lower}@ticker"
        elif data_type == DataType.TRADES:
            stream_name = f"{symbol_lower}@trade"
        elif data_type == DataType.ORDERBOOK:
            stream_name = f"{symbol_lower}@depth@100ms"
        elif data_type == DataType.KLINES:
            stream_name = f"{symbol_lower}@kline_1m"
        elif data_type == DataType.DEPTH:
            stream_name = f"{symbol_lower}@depth"
        
        # Envoyer la commande de souscription
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": int(time.time() * 1000)
        }
        
        try:
            await self.websocket.send(json.dumps(subscribe_message))
            log_debug(f"Souscrit au stream: {stream_name}")
        except Exception as e:
            log_error(f"Erreur lors de la souscription à {stream_name}: {e}")

    async def _process_message(self, data: Dict, receive_time: float):
        """
        Traite un message WebSocket
        
        Args:
            data: Données reçues
            receive_time: Timestamp de réception
        """
        try:
            # Identifier le type de message
            event_type = data.get('e')
            if not event_type:
                return
            
            symbol = data.get('s', '').lower()
            
            # Calculer la latence
            if 'E' in data:  # Event time de Binance
                event_time = data['E'] / 1000.0  # Convertir en secondes
                latency_ms = (receive_time - event_time) * 1000
                self.latency_buffer.append(latency_ms)
            else:
                latency_ms = 0
            
            # Déterminer le type de données
            data_type = None
            processed_data = {}
            
            if event_type == '24hrTicker':
                data_type = DataType.TICKER
                processed_data = {
                    'bid': float(data.get('b', 0)),
                    'ask': float(data.get('a', 0)),
                    'last': float(data.get('c', 0)),
                    'volume': float(data.get('v', 0)),
                    'quote_volume': float(data.get('q', 0)),
                    'change_24h': float(data.get('P', 0))
                }
                
            elif event_type == 'trade':
                data_type = DataType.TRADES
                processed_data = {
                    'price': float(data.get('p', 0)),
                    'quantity': float(data.get('q', 0)),
                    'time': data.get('T', 0),
                    'is_buyer_maker': data.get('m', False)
                }
                
            elif event_type == 'depthUpdate':
                data_type = DataType.DEPTH
                processed_data = {
                    'bids': [[float(p), float(q)] for p, q in data.get('b', [])],
                    'asks': [[float(p), float(q)] for p, q in data.get('a', [])],
                    'update_id': data.get('u', 0)
                }
                
            elif event_type == 'kline':
                data_type = DataType.KLINES
                kline = data.get('k', {})
                processed_data = {
                    'time': kline.get('t', 0),
                    'open': float(kline.get('o', 0)),
                    'high': float(kline.get('h', 0)),
                    'low': float(kline.get('l', 0)),
                    'close': float(kline.get('c', 0)),
                    'volume': float(kline.get('v', 0)),
                    'closed': kline.get('x', False)
                }
            
            if data_type and symbol:
                # Créer l'update
                market_update = MarketUpdate(
                    symbol=symbol,
                    data_type=data_type,
                    data=processed_data,
                    timestamp=receive_time,
                    latency_ms=latency_ms,
                    exchange=self.exchange
                )
                
                # Mettre à jour le cache
                self._update_cache(market_update)
                
                # Appeler les callbacks
                await self._dispatch_callbacks(symbol, market_update)
                
                # Suivi latence uniquement via LatencyMonitor (plus de log direct)
                if latency_ms > 200:
                    self.latency_monitor.add_latency(symbol, latency_ms)
        
        except Exception as e:
            log_error(f"Erreur traitement message: {e}")
            self.error_count += 1
    
    def _update_cache(self, update: MarketUpdate):
        """Met à jour le cache local pour accès rapide"""
        symbol = update.symbol
        
        if update.data_type == DataType.TICKER:
            self.ticker_cache[symbol] = update.data
        
        elif update.data_type == DataType.DEPTH:
            # Mise à jour incrémentale du carnet d'ordres
            if symbol in self.orderbook_cache:
                snapshot = self.orderbook_cache[symbol]
                
                # Appliquer les mises à jour
                for bid in update.data['bids']:
                    price, qty = bid
                    if qty == 0:
                        # Supprimer le niveau
                        snapshot.bids = [b for b in snapshot.bids if b[0] != price]
                    else:
                        # Mettre à jour ou ajouter
                        updated = False
                        for i, (p, _) in enumerate(snapshot.bids):
                            if p == price:
                                snapshot.bids[i] = [price, qty]
                                updated = True
                                break
                        if not updated:
                            snapshot.bids.append([price, qty])
                
                # Même chose pour les asks
                for ask in update.data['asks']:
                    price, qty = ask
                    if qty == 0:
                        snapshot.asks = [a for a in snapshot.asks if a[0] != price]
                    else:
                        updated = False
                        for i, (p, _) in enumerate(snapshot.asks):
                            if p == price:
                                snapshot.asks[i] = [price, qty]
                                updated = True
                                break
                        if not updated:
                            snapshot.asks.append([price, qty])
                
                # Trier et limiter
                snapshot.bids.sort(key=lambda x: x[0], reverse=True)
                snapshot.asks.sort(key=lambda x: x[0])
                snapshot.bids = snapshot.bids[:100]
                snapshot.asks = snapshot.asks[:100]
                snapshot.timestamp = update.timestamp
                snapshot.update_id = update.data.get('update_id', snapshot.update_id)

    async def _dispatch_callbacks(self, symbol: str, update: MarketUpdate):
        """Appelle les callbacks enregistrés"""
        # Callbacks spécifiques au symbole
        if symbol in self.callbacks:
            for callback in self.callbacks[symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update)
                    else:
                        callback(update)
                except Exception as e:
                    log_error(f"Erreur dans callback: {e}")
        
        # Callbacks globaux (wildcard)
        if '*' in self.callbacks:
            for callback in self.callbacks['*']:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update)
                    else:
                        callback(update)
                except Exception as e:
                    log_error(f"Erreur dans callback global: {e}")
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Récupère le dernier ticker du cache
        
        Args:
            symbol: Symbole
        
        Returns:
            Données du ticker ou None
        """
        normalized = symbol.replace('/', '').lower()
        return self.ticker_cache.get(normalized)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """
        Récupère le dernier orderbook du cache
        
        Args:
            symbol: Symbole
        
        Returns:
            Snapshot de l'orderbook ou None
        """
        normalized = symbol.replace('/', '').lower()
        return self.orderbook_cache.get(normalized)
    
    def get_metrics(self) -> Dict:
        """Retourne les métriques de performance"""
        uptime = time.time() - self.start_time
        avg_latency = sum(self.latency_buffer) / len(self.latency_buffer) if self.latency_buffer else 0
        
        return {
            'connected': self.connected,
            'uptime_seconds': uptime,
            'message_count': self.message_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.message_count, 1),
            'avg_latency_ms': avg_latency,
            'min_latency_ms': min(self.latency_buffer) if self.latency_buffer else 0,
            'max_latency_ms': max(self.latency_buffer) if self.latency_buffer else 0,
            'subscriptions': len(self.subscriptions),
            'cached_tickers': len(self.ticker_cache),
            'cached_orderbooks': len(self.orderbook_cache)
        }

    def subscribe(self, symbol: str, data_types: List[DataType], 
                  callback: Optional[Callable] = None):
        """
        S'abonne aux données d'un symbole
        
        Args:
            symbol: Symbole (ex: BTC/USDT)
            data_types: Types de données à recevoir
            callback: Fonction appelée lors des mises à jour
        """
        # Normaliser le symbole
        symbol_normalized = symbol.replace('/', '').lower()
        
        # Ajouter les souscriptions
        for data_type in data_types:
            self.subscriptions[symbol_normalized].add(data_type)
            
            # Si déjà connecté, ajouter à la file d'attente
            if self.connected and self.websocket:
                self.pending_subscriptions.append((symbol, data_type))
                if not self._subscription_task or self._subscription_task.done():
                    self._subscription_task = asyncio.create_task(self._process_subscriptions())
        
        # Ajouter le callback si fourni
        if callback:
            self.callbacks[symbol_normalized].append(callback)
        
        log_info(f"Souscription ajoutée: {symbol} - {[dt.value for dt in data_types]}")