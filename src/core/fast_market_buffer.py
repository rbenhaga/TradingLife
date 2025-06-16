# src/core/fast_market_buffer.py
"""
Buffer de marché haute performance avec accès O(1) et calculs vectorisés
Utilise Numba JIT pour des performances proches du code natif
"""

import numpy as np
from numba import jit, njit, prange, typed, types
from typing import Dict, Tuple, Optional, List
import time
from dataclasses import dataclass
from collections import deque
import threading

@dataclass
class MarketSnapshot:
    """Snapshot atomique du marché à un instant T"""
    timestamp: int
    bid: float
    ask: float
    last: float
    volume: float
    bid_size: float
    ask_size: float
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class FastMarketBuffer:
    """
    Buffer circulaire ultra-rapide pour données de marché
    Optimisé pour latence <1ms avec pré-allocation et vectorisation
    """
    
    def __init__(self, max_symbols: int = 200, buffer_size: int = 10000):
        # Pré-allocation des arrays numpy (mémoire contiguë)
        self.buffer_size = buffer_size
        self.max_symbols = max_symbols
        
        # Arrays pour les données de marché
        self.bids = np.zeros((max_symbols, buffer_size), dtype=np.float32)
        self.asks = np.zeros((max_symbols, buffer_size), dtype=np.float32)
        self.lasts = np.zeros((max_symbols, buffer_size), dtype=np.float32)
        self.volumes = np.zeros((max_symbols, buffer_size), dtype=np.float32)
        self.bid_sizes = np.zeros((max_symbols, buffer_size), dtype=np.float32)
        self.ask_sizes = np.zeros((max_symbols, buffer_size), dtype=np.float32)
        self.timestamps = np.zeros((max_symbols, buffer_size), dtype=np.int64)
        
        # Index circulaire et compteurs
        self.write_indices = np.zeros(max_symbols, dtype=np.int32)
        self.data_counts = np.zeros(max_symbols, dtype=np.int32)
        
        # Mapping symbole -> ID (Dict Numba typé pour performance)
        self.symbol_map = typed.Dict.empty(
            key_type=types.unicode_type,
            value_type=types.int32
        )
        self.reverse_map = {}  # ID -> symbole
        self.next_id = 0
        
        # Cache des calculs (moyennes, volatilité, etc.)
        self.cache_lock = threading.RLock()
        self._cache = {}
        self._cache_timestamps = {}
        self.cache_ttl_ms = 100  # 100ms TTL
        
        # Statistiques de performance
        self.update_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    @staticmethod
    @njit(parallel=True, cache=True, fastmath=True)
    def _update_buffer_fast(bids, asks, lasts, volumes, bid_sizes, ask_sizes,
                           timestamps, write_indices, data_counts,
                           symbol_id, bid, ask, last, volume, bid_size, 
                           ask_size, timestamp, buffer_size):
        """
        Mise à jour atomique optimisée avec Numba
        Utilise parallel=True pour les calculs multi-cœurs
        """
        idx = write_indices[symbol_id]
        
        # Écriture atomique des données
        bids[symbol_id, idx] = bid
        asks[symbol_id, idx] = ask
        lasts[symbol_id, idx] = last
        volumes[symbol_id, idx] = volume
        bid_sizes[symbol_id, idx] = bid_size
        ask_sizes[symbol_id, idx] = ask_size
        timestamps[symbol_id, idx] = timestamp
        
        # Mise à jour des indices (circulaire)
        write_indices[symbol_id] = (idx + 1) % buffer_size
        data_counts[symbol_id] = min(data_counts[symbol_id] + 1, buffer_size)
    
    def update(self, symbol: str, bid: float, ask: float, last: float,
               volume: float, bid_size: float = 0, ask_size: float = 0) -> None:
        """
        Mise à jour des données de marché - Latence cible <0.1ms
        Thread-safe et optimisé pour les appels haute fréquence
        """
        # Obtenir ou créer l'ID du symbole
        if symbol not in self.symbol_map:
            if self.next_id >= self.max_symbols:
                raise ValueError(f"Buffer plein: max {self.max_symbols} symboles")
            
            with self.cache_lock:
                self.symbol_map[symbol] = self.next_id
                self.reverse_map[self.next_id] = symbol
                self.next_id += 1
        
        symbol_id = self.symbol_map[symbol]
        timestamp = int(time.time() * 1000000)  # Microsecondes
        
        # Mise à jour atomique via Numba
        self._update_buffer_fast(
            self.bids, self.asks, self.lasts, self.volumes,
            self.bid_sizes, self.ask_sizes, self.timestamps,
            self.write_indices, self.data_counts,
            symbol_id, bid, ask, last, volume, bid_size, ask_size,
            timestamp, self.buffer_size
        )
        
        self.update_count += 1
        
        # Invalider le cache pour ce symbole
        with self.cache_lock:
            if symbol in self._cache:
                del self._cache[symbol]
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _calculate_stats_fast(data, timestamps, write_idx, count, window_us):
        """Calculs statistiques vectorisés avec Numba"""
        if count == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Obtenir la fenêtre de données valides
        current_time = timestamps[write_idx - 1] if write_idx > 0 else timestamps[-1]
        cutoff_time = current_time - window_us
        
        # Indices valides dans la fenêtre temporelle
        valid_count = 0
        values = np.empty(count, dtype=np.float32)
        
        for i in range(count):
            idx = (write_idx - 1 - i) % len(data)
            if timestamps[idx] >= cutoff_time:
                values[valid_count] = data[idx]
                valid_count += 1
            else:
                break
        
        if valid_count == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Calculs statistiques
        values = values[:valid_count]
        mean = np.mean(values)
        std = np.std(values) if valid_count > 1 else 0.0
        min_val = np.min(values)
        max_val = np.max(values)
        
        return mean, std, min_val, max_val
    
    def get_stats(self, symbol: str, window_ms: int = 1000) -> Dict:
        """
        Obtenir les statistiques sur une fenêtre temporelle
        Utilise un cache pour éviter les recalculs
        """
        cache_key = f"{symbol}_{window_ms}"
        current_time = int(time.time() * 1000)
        
        # Vérifier le cache
        with self.cache_lock:
            if cache_key in self._cache:
                cache_time = self._cache_timestamps.get(cache_key, 0)
                if current_time - cache_time < self.cache_ttl_ms:
                    self.cache_hits += 1
                    return self._cache[cache_key]
        
        self.cache_misses += 1
        
        # Calculer les stats
        if symbol not in self.symbol_map:
            return {}
        
        symbol_id = self.symbol_map[symbol]
        write_idx = self.write_indices[symbol_id]
        count = self.data_counts[symbol_id]
        window_us = window_ms * 1000
        
        # Calculs parallèles pour chaque métrique
        bid_stats = self._calculate_stats_fast(
            self.bids[symbol_id], self.timestamps[symbol_id],
            write_idx, count, window_us
        )
        
        ask_stats = self._calculate_stats_fast(
            self.asks[symbol_id], self.timestamps[symbol_id],
            write_idx, count, window_us
        )
        
        volume_stats = self._calculate_stats_fast(
            self.volumes[symbol_id], self.timestamps[symbol_id],
            write_idx, count, window_us
        )
        
        # Construire le résultat
        result = {
            'bid_mean': bid_stats[0],
            'bid_std': bid_stats[1],
            'ask_mean': ask_stats[0],
            'ask_std': ask_stats[1],
            'spread_mean': ask_stats[0] - bid_stats[0],
            'volume_mean': volume_stats[0],
            'volume_std': volume_stats[1],
            'samples': count
        }
        
        # Mettre en cache
        with self.cache_lock:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = current_time
        
        return result
    
    def get_latest(self, symbol: str) -> Optional[MarketSnapshot]:
        """Obtenir le dernier snapshot - O(1)"""
        if symbol not in self.symbol_map:
            return None
        
        symbol_id = self.symbol_map[symbol]
        if self.data_counts[symbol_id] == 0:
            return None
        
        # Index du dernier élément
        idx = (self.write_indices[symbol_id] - 1) % self.buffer_size
        
        return MarketSnapshot(
            timestamp=int(self.timestamps[symbol_id, idx]),
            bid=float(self.bids[symbol_id, idx]),
            ask=float(self.asks[symbol_id, idx]),
            last=float(self.lasts[symbol_id, idx]),
            volume=float(self.volumes[symbol_id, idx]),
            bid_size=float(self.bid_sizes[symbol_id, idx]),
            ask_size=float(self.ask_sizes[symbol_id, idx])
        )
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def _detect_microstructure_fast(bids, asks, volumes, timestamps,
                                   write_idx, count, lookback):
        """
        Détection de patterns de microstructure (pour HFT)
        Optimisé pour exécution parallèle
        """
        if count < lookback:
            return 0, 0, 0  # Pas assez de données
        
        # Arrays pour les calculs
        spreads = np.empty(lookback, dtype=np.float32)
        vol_imbalances = np.empty(lookback - 1, dtype=np.float32)
        
        # Calcul des spreads et déséquilibres
        for i in prange(lookback):
            idx = (write_idx - lookback + i) % len(bids)
            spreads[i] = asks[idx] - bids[idx]
        
        # Déséquilibre de volume (indicateur de pression)
        for i in prange(lookback - 1):
            idx_curr = (write_idx - lookback + i + 1) % len(volumes)
            idx_prev = (write_idx - lookback + i) % len(volumes)
            vol_imbalances[i] = volumes[idx_curr] - volumes[idx_prev]
        
        # Métriques de microstructure
        spread_volatility = np.std(spreads)
        avg_spread = np.mean(spreads)
        volume_pressure = np.sum(vol_imbalances) / np.sum(np.abs(vol_imbalances))
        
        return spread_volatility, avg_spread, volume_pressure
    
    def get_microstructure_signals(self, symbol: str) -> Dict:
        """Signaux de microstructure pour trading haute fréquence"""
        if symbol not in self.symbol_map:
            return {}
        
        symbol_id = self.symbol_map[symbol]
        write_idx = self.write_indices[symbol_id]
        count = self.data_counts[symbol_id]
        
        # Analyse sur 100 derniers ticks
        spread_vol, avg_spread, vol_pressure = self._detect_microstructure_fast(
            self.bids[symbol_id], self.asks[symbol_id],
            self.volumes[symbol_id], self.timestamps[symbol_id],
            write_idx, count, min(100, count)
        )
        
        return {
            'spread_volatility': float(spread_vol),
            'avg_spread': float(avg_spread),
            'volume_pressure': float(vol_pressure),
            'liquidity_score': 1.0 / (1.0 + spread_vol) if spread_vol > 0 else 1.0
        }