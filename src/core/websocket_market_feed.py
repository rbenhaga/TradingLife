"""
Module WebSocket pour données de marché temps réel haute performance
Conforme aux exigences de latence < 100ms
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
    BINANCE_WS_TESTNET = "wss://testnet.binance.vision/ws"
    
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
        self.ws_url = self.BINANCE_WS_TESTNET if testnet else self.BINANCE_WS_BASE
        
        # État de connexion
        self.websocket = None
        self.connected = False
        self.reconnect_count = 0
        
        # Subscriptions
        self.subscriptions: Dict[str, Set[DataType]] = defaultdict(set)
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Métriques de performance
        self.latency_buffer = deque(maxlen=1000)  # Dernières 1000 latences
        self.message_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Cache pour réduire la latence
        self.ticker_cache: Dict[str, Dict] = {}
        self.orderbook_cache: Dict[str, OrderBookSnapshot] = {}

        log_info(f"WebSocketMarketFeed initialisé - Exchange: {exchange}, Testnet: {testnet}")

    async def fetch_snapshot(self, symbol: str, limit: int = 200) -> Optional[OrderBookSnapshot]:
        """Récupère un snapshot initial du carnet d'ordres via l'API REST."""
        base = "https://testnet.binance.vision" if self.testnet else "https://api.binance.com"
        url = f"{base}/api/v3/depth"
        params = {"symbol": symbol.upper(), "limit": limit}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
            asks = [[float(p), float(q)] for p, q in data.get("asks", [])]
            snapshot = OrderBookSnapshot(
                symbol=symbol.lower(),
                bids=bids[:limit],
                asks=asks[:limit],
                timestamp=time.time(),
                update_id=data.get("lastUpdateId", 0),
            )
            self.orderbook_cache[symbol.lower()] = snapshot
            log_info(f"Snapshot récupéré pour {symbol} (update_id={snapshot.update_id})")
            return snapshot
        except Exception as e:
            log_error(f"Erreur fetch_snapshot pour {symbol}: {e}")
            return None

    async def _initialize_orderbook_snapshots(self):
        """Initialise les snapshots pour tous les symboles souscrits."""
        tasks = []
        for symbol, types in self.subscriptions.items():
            if DataType.ORDERBOOK in types or DataType.DEPTH in types:
                tasks.append(self.fetch_snapshot(symbol))

        if tasks:
            await asyncio.gather(*tasks)
    
    async def connect(self) -> bool:
        """
        Établit la connexion WebSocket
        
        Returns:
            True si connecté avec succès
        """
        try:
            # Construire l'URL avec les streams
            streams = self._build_stream_list()
            if streams:
                url = f"{self.ws_url}/{'/'.join(streams)}"
            else:
                url = self.ws_url
            
            log_info(f"Connexion WebSocket à: {url}")
            
            # Connecter avec timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(url, ping_interval=20),
                timeout=10.0
            )
            
            self.connected = True
            self.reconnect_count = 0

            log_info("✅ WebSocket connecté avec succès")

            # Récupérer les snapshots initiaux pour les carnets d'ordres
            await self._initialize_orderbook_snapshots()

            # Démarrer la boucle de réception
            asyncio.create_task(self._receive_loop())

            # Démarrer le heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            return True
            
        except asyncio.TimeoutError:
            log_error("Timeout lors de la connexion WebSocket")
            return False
        except Exception as e:
            log_error(f"Erreur de connexion WebSocket: {e}")
            return False
    
    async def disconnect(self):
        """Ferme la connexion WebSocket"""
        self.connected = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        log_info("WebSocket déconnecté")
    
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
        
        # Ajouter le callback si fourni
        if callback:
            self.callbacks[symbol_normalized].append(callback)
        
        log_info(f"Souscription ajoutée: {symbol} - {[dt.value for dt in data_types]}")
        
        # Si déjà connecté, mettre à jour la souscription
        if self.connected:
            asyncio.create_task(self._update_subscriptions())
    
    def unsubscribe(self, symbol: str, data_types: Optional[List[DataType]] = None):
        """
        Se désabonne des données d'un symbole
        
        Args:
            symbol: Symbole
            data_types: Types spécifiques (None = tout)
        """
        symbol_normalized = symbol.replace('/', '').lower()
        
        if data_types:
            for data_type in data_types:
                self.subscriptions[symbol_normalized].discard(data_type)
        else:
            # Désabonner de tout
            self.subscriptions[symbol_normalized].clear()
        
        # Nettoyer si vide
        if not self.subscriptions[symbol_normalized]:
            del self.subscriptions[symbol_normalized]
            if symbol_normalized in self.callbacks:
                del self.callbacks[symbol_normalized]
    
    def _build_stream_list(self) -> List[str]:
        """Construit la liste des streams pour l'URL"""
        streams = []
        
        for symbol, data_types in self.subscriptions.items():
            for data_type in data_types:
                if data_type == DataType.TICKER:
                    streams.append(f"{symbol}@ticker")
                elif data_type == DataType.TRADES:
                    streams.append(f"{symbol}@trade")
                elif data_type == DataType.ORDERBOOK:
                    streams.append(f"{symbol}@depth@100ms")
                elif data_type == DataType.KLINES:
                    streams.append(f"{symbol}@kline_1m")
                elif data_type == DataType.DEPTH:
                    streams.append(f"{symbol}@depth")
        
        return streams
    
    async def _receive_loop(self):
        """Boucle de réception des messages WebSocket"""
        while self.connected:
            try:
                # Recevoir le message
                message = await self.websocket.recv()
                receive_time = time.time()
                
                # Parser le JSON
                data = json.loads(message)
                
                # Traiter le message
                await self._process_message(data, receive_time)
                
                self.message_count += 1
                
            except websockets.ConnectionClosed:
                log_warning("Connexion WebSocket fermée")
                await self._handle_disconnect()
                break
            except json.JSONDecodeError as e:
                log_error(f"Erreur de parsing JSON: {e}")
                self.error_count += 1
            except Exception as e:
                log_error(f"Erreur dans receive_loop: {e}")
                self.error_count += 1
                await asyncio.sleep(0.1)
    
    async def _process_message(self, data: Dict, receive_time: float):
        """
        Traite un message WebSocket
        
        Args:
            data: Données reçues
            receive_time: Timestamp de réception
        """
        try:
            # Identifier le type de stream
            if 'stream' in data:
                stream = data['stream']
                payload = data['data']
            else:
                stream = self._identify_stream(data)
                payload = data
            
            if not stream:
                return
            
            # Extraire symbol et type
            parts = stream.split('@')
            symbol = parts[0]
            stream_type = parts[1] if len(parts) > 1 else None
            
            # Calculer la latence
            if 'E' in payload:  # Event time de Binance
                event_time = payload['E'] / 1000.0  # Convertir en secondes
                latency_ms = (receive_time - event_time) * 1000
                self.latency_buffer.append(latency_ms)
            else:
                latency_ms = 0
            
            # Créer l'update
            market_update = self._create_market_update(
                symbol, stream_type, payload, receive_time, latency_ms
            )
            
            if market_update:
                # Mettre à jour le cache
                self._update_cache(market_update)
                
                # Appeler les callbacks
                await self._dispatch_callbacks(symbol, market_update)
                
                # Log si latence élevée
                if latency_ms > 100:
                    log_warning(f"Latence élevée détectée: {latency_ms:.1f}ms pour {symbol}")
        
        except Exception as e:
            log_error(f"Erreur traitement message: {e}")
            self.error_count += 1
    
    def _identify_stream(self, data: Dict) -> Optional[str]:
        """Identifie le type de stream depuis les données"""
        # Logique spécifique à Binance
        if 'e' in data:
            event_type = data['e']
            symbol = data.get('s', '').lower()
            
            if event_type == '24hrTicker':
                return f"{symbol}@ticker"
            elif event_type == 'trade':
                return f"{symbol}@trade"
            elif event_type == 'depthUpdate':
                return f"{symbol}@depth"
            elif event_type == 'kline':
                return f"{symbol}@kline_{data.get('k', {}).get('i', '1m')}"
        
        return None
    
    def _create_market_update(self, symbol: str, stream_type: str, 
                            data: Dict, timestamp: float, 
                            latency_ms: float) -> Optional[MarketUpdate]:
        """Crée un MarketUpdate depuis les données"""
        try:
            if stream_type == 'ticker':
                return MarketUpdate(
                    symbol=symbol,
                    data_type=DataType.TICKER,
                    data={
                        'bid': float(data.get('b', 0)),
                        'ask': float(data.get('a', 0)),
                        'last': float(data.get('c', 0)),
                        'volume': float(data.get('v', 0)),
                        'quote_volume': float(data.get('q', 0)),
                        'change_24h': float(data.get('P', 0))
                    },
                    timestamp=timestamp,
                    latency_ms=latency_ms,
                    exchange=self.exchange
                )
            
            elif stream_type == 'trade':
                return MarketUpdate(
                    symbol=symbol,
                    data_type=DataType.TRADES,
                    data={
                        'price': float(data.get('p', 0)),
                        'quantity': float(data.get('q', 0)),
                        'time': data.get('T', 0),
                        'is_buyer_maker': data.get('m', False)
                    },
                    timestamp=timestamp,
                    latency_ms=latency_ms,
                    exchange=self.exchange
                )
            
            elif stream_type and stream_type.startswith('depth'):
                # Mise à jour ou snapshot du carnet d'ordres
                bids_raw = data.get('bids') or data.get('b') or []
                asks_raw = data.get('asks') or data.get('a') or []
                bids = [[float(p), float(q)] for p, q in bids_raw]
                asks = [[float(p), float(q)] for p, q in asks_raw]

                return MarketUpdate(
                    symbol=symbol,
                    data_type=DataType.ORDERBOOK,
                    data={
                        'bids': bids[:200],
                        'asks': asks[:200],
                        'update_id': data.get('u') or data.get('lastUpdateId', 0)
                    },
                    timestamp=timestamp,
                    latency_ms=latency_ms,
                    exchange=self.exchange
                )
            
            elif stream_type and stream_type.startswith('kline'):
                kline = data.get('k', {})
                return MarketUpdate(
                    symbol=symbol,
                    data_type=DataType.KLINES,
                    data={
                        'time': kline.get('t', 0),
                        'open': float(kline.get('o', 0)),
                        'high': float(kline.get('h', 0)),
                        'low': float(kline.get('l', 0)),
                        'close': float(kline.get('c', 0)),
                        'volume': float(kline.get('v', 0)),
                        'closed': kline.get('x', False)
                    },
                    timestamp=timestamp,
                    latency_ms=latency_ms,
                    exchange=self.exchange
                )
            
        except Exception as e:
            log_error(f"Erreur création MarketUpdate: {e}")
        
        return None
    
    def _update_cache(self, update: MarketUpdate):
        """Met à jour le cache local pour accès rapide"""
        symbol = update.symbol
        
        if update.data_type == DataType.TICKER:
            self.ticker_cache[symbol] = update.data
        
        elif update.data_type == DataType.ORDERBOOK:
            if symbol in self.orderbook_cache:
                self._merge_orderbook(self.orderbook_cache[symbol], update)
            else:
                self.orderbook_cache[symbol] = OrderBookSnapshot(
                    symbol=symbol,
                    bids=update.data['bids'],
                    asks=update.data['asks'],
                    timestamp=update.timestamp,
                    update_id=update.data.get('update_id', 0)
                )

    def _merge_orderbook(self, snapshot: OrderBookSnapshot, update: MarketUpdate):
        """Fusionne une mise à jour de carnet avec le snapshot existant."""
        def apply(levels: List[List[float]], changes: List[List[float]], reverse: bool) -> List[List[float]]:
            book = {price: qty for price, qty in levels}
            for price, qty in changes:
                if qty == 0:
                    book.pop(price, None)
                else:
                    book[price] = qty
            ordered = sorted(book.items(), key=lambda x: x[0], reverse=reverse)
            return [[p, q] for p, q in ordered[:200]]

        snapshot.bids = apply(snapshot.bids, update.data['bids'], True)
        snapshot.asks = apply(snapshot.asks, update.data['asks'], False)
        snapshot.update_id = update.data.get('update_id', snapshot.update_id)
        snapshot.timestamp = update.timestamp

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
    
    async def _handle_disconnect(self):
        """Gère la déconnexion et tente de reconnecter"""
        self.connected = False
        
        if self.reconnect_count < self.max_reconnect_attempts:
            self.reconnect_count += 1
            wait_time = min(2 ** self.reconnect_count, 60)  # Backoff exponentiel
            
            log_warning(f"Tentative de reconnexion {self.reconnect_count}/{self.max_reconnect_attempts} dans {wait_time}s")
            await asyncio.sleep(wait_time)
            
            # Tenter de reconnecter
            if await self.connect():
                log_info("Reconnexion réussie")
            else:
                await self._handle_disconnect()
        else:
            log_error("Nombre maximum de tentatives de reconnexion atteint")
    
    async def _heartbeat_loop(self):
        """Envoie des pings périodiques pour maintenir la connexion"""
        while self.connected:
            try:
                await asyncio.sleep(30)  # Ping toutes les 30 secondes
                if self.websocket:
                    pong = await self.websocket.ping()
                    await pong
            except Exception as e:
                log_error(f"Erreur heartbeat: {e}")
                await self._handle_disconnect()
                break
    
    async def _update_subscriptions(self):
        """Met à jour les souscriptions (reconnexion nécessaire avec Binance)"""
        # Binance nécessite une reconnexion pour changer les streams
        await self.disconnect()
        await self.connect()
    
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


# Classe d'agrégation pour gérer plusieurs feeds
class MultiExchangeFeed:
    """
    Agrégateur de feeds multi-exchanges
    """
    
    def __init__(self):
        """Initialise l'agrégateur"""
        self.feeds: Dict[str, WebSocketMarketFeed] = {}
        self.unified_callbacks: List[Callable] = []
    
    async def add_exchange(self, exchange: str, testnet: bool = False) -> bool:
        """Ajoute un exchange"""
        if exchange not in self.feeds:
            feed = WebSocketMarketFeed(exchange, testnet)
            if await feed.connect():
                self.feeds[exchange] = feed
                return True
        return False
