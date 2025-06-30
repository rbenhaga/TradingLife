# src/core/fast_market_buffer.py
"""
Buffer de marché haute performance avec optimisations Numba
Objectif: < 0.1ms pour update, < 1ms pour stats
"""

import numpy as np
from numba import njit, jit, prange, types, int64, float64
from numba.typed import Dict as NumbaDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import time
from collections import deque
import threading


@dataclass
class MarketSnapshot:
    """Snapshot immutable des données de marché"""
    timestamp: int  # Unix timestamp en ms
    bid: float
    ask: float
    last: float
    volume: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2


class FastMarketBuffer:
    """
    Buffer circulaire haute performance pour données de marché
    Utilise Numba pour les calculs critiques
    """
    
    def __init__(self, max_symbols: int = 100, buffer_size: int = 10000):
        """
        Initialise le buffer avec pré-allocation
        
        Args:
            max_symbols: Nombre maximum de symboles
            buffer_size: Taille du buffer circulaire par symbole
        """
        self.max_symbols = max_symbols
        self.buffer_size = buffer_size
        
        # Pré-allocation des arrays numpy (colonnes = symboles, lignes = temps)
        self.timestamps = np.zeros((max_symbols, buffer_size), dtype=np.int64)
        self.bids = np.zeros((max_symbols, buffer_size), dtype=np.float64)
        self.asks = np.zeros((max_symbols, buffer_size), dtype=np.float64)
        self.lasts = np.zeros((max_symbols, buffer_size), dtype=np.float64)
        self.volumes = np.zeros((max_symbols, buffer_size), dtype=np.float64)
        self.bid_sizes = np.zeros((max_symbols, buffer_size), dtype=np.float64)
        self.ask_sizes = np.zeros((max_symbols, buffer_size), dtype=np.float64)
        
        # Indices circulaires et compteurs (utiliser int64 pour éviter les warnings)
        self.write_indices = np.zeros(max_symbols, dtype=np.int64)
        self.data_counts = np.zeros(max_symbols, dtype=np.int64)
        
        # Mapping symbole -> index
        self.symbol_map: Dict[str, int] = {}
        self.next_symbol_id = 0
        
        # Lock pour thread-safety
        self.lock = threading.RLock()
        
        # Cache pour les calculs fréquents
        self._cache = {}
        self._cache_timestamps = {}
        self.cache_ttl_ms = 100  # 100ms TTL
        
        # Stats de performance
        self.update_latencies = deque(maxlen=1000)
        self.stats_latencies = deque(maxlen=1000)
    
    def _get_or_create_symbol_id(self, symbol: str) -> Optional[int]:
        """Obtient ou crée un ID pour le symbole"""
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        
        with self.lock:
            # Double-check après acquisition du lock
            if symbol in self.symbol_map:
                return self.symbol_map[symbol]
            
            if self.next_symbol_id >= self.max_symbols:
                return None  # Buffer plein
            
            symbol_id = self.next_symbol_id
            self.symbol_map[symbol] = symbol_id
            self.next_symbol_id += 1
            return symbol_id
    
    def update(self, symbol: str, bid: float, ask: float, last: float,
               volume: float, bid_size: float = 0.0, ask_size: float = 0.0,
               timestamp: Optional[int] = None) -> bool:
        """
        Met à jour les données pour un symbole - O(1)
        
        Returns:
            True si succès, False sinon
        """
        start_time = time.perf_counter()
        
        # Obtenir l'ID du symbole
        symbol_id = self._get_or_create_symbol_id(symbol)
        if symbol_id is None:
            return False
        
        # Timestamp en millisecondes
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        
        # Mise à jour atomique avec Numba
        idx = self._update_buffer_fast(
            symbol_id,
            timestamp,
            bid, ask, last, volume,
            bid_size, ask_size,
            self.write_indices,
            self.data_counts,
            self.timestamps,
            self.bids, self.asks, self.lasts,
            self.volumes, self.bid_sizes, self.ask_sizes,
            self.buffer_size
        )
        
        # Invalider le cache pour ce symbole
        with self.lock:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{symbol}:")]
            for k in keys_to_remove:
                del self._cache[k]
                del self._cache_timestamps[k]
        
        # Enregistrer la latence
        latency = (time.perf_counter() - start_time) * 1000  # ms
        self.update_latencies.append(latency)
        
        return True
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _update_buffer_fast(symbol_id, timestamp,
                           bid, ask, last, volume,
                           bid_size, ask_size,
                           write_indices, data_counts,
                           timestamps, bids, asks, lasts,
                           volumes, bid_sizes, ask_sizes, buffer_size):
        """
        Mise à jour atomique du buffer avec Numba
        """
        # Obtenir l'index d'écriture
        idx = write_indices[symbol_id]
        
        # Écrire les données
        timestamps[symbol_id, idx] = timestamp
        bids[symbol_id, idx] = bid
        asks[symbol_id, idx] = ask
        lasts[symbol_id, idx] = last
        volumes[symbol_id, idx] = volume
        bid_sizes[symbol_id, idx] = bid_size
        ask_sizes[symbol_id, idx] = ask_size
        
        # Incrémenter l'index circulaire
        write_indices[symbol_id] = (idx + 1) % buffer_size
        
        # Incrémenter le compteur (max = buffer_size)
        if data_counts[symbol_id] < buffer_size:
            data_counts[symbol_id] += 1
        
        return idx
    
    def get_stats(self, symbol: str, window_ms: int = 1000,
                  use_cache: bool = True) -> Optional[Dict[str, float]]:
        """
        Calcule les statistiques sur une fenêtre temporelle - O(n)
        
        Args:
            symbol: Symbole à analyser
            window_ms: Fenêtre en millisecondes
            use_cache: Utiliser le cache si disponible
        
        Returns:
            Dict avec mean, std, min, max pour bid/ask/spread/volume
        """
        start_time = time.perf_counter()
        
        # Vérifier le cache
        cache_key = f"{symbol}:{window_ms}"
        if use_cache:
            with self.lock:
                if cache_key in self._cache:
                    cache_time = self._cache_timestamps[cache_key]
                    if (time.time() * 1000 - cache_time) < self.cache_ttl_ms:
                        return self._cache[cache_key]
        
        # Obtenir l'ID du symbole
        if symbol not in self.symbol_map:
            return None
        
        symbol_id = self.symbol_map[symbol]
        count = self.data_counts[symbol_id]
        
        if count == 0:
            return None
        
        # Calculer avec Numba
        current_time = int(time.time() * 1000)
        stats = self._calculate_stats_fast(
            symbol_id,
            current_time,
            window_ms,
            self.write_indices[symbol_id],
            count,
            self.timestamps[symbol_id],
            self.bids[symbol_id],
            self.asks[symbol_id],
            self.volumes[symbol_id],
            self.buffer_size
        )
        
        # Convertir en dict Python
        result = {
            'samples': int(stats[0]),
            'bid': {
                'mean': float(stats[1]),
                'std': float(stats[2]),
                'min': float(stats[3]),
                'max': float(stats[4])
            },
            'ask': {
                'mean': float(stats[5]),
                'std': float(stats[6]),
                'min': float(stats[7]),
                'max': float(stats[8])
            },
            'spread': {
                'mean': float(stats[9]),
                'std': float(stats[10]),
                'min': float(stats[11]),
                'max': float(stats[12])
            },
            'volume': {
                'mean': float(stats[13]),
                'std': float(stats[14]),
                'total': float(stats[15]),
                'current': float(stats[16])
            }
        }
        
        # Mettre en cache
        if use_cache:
            with self.lock:
                self._cache[cache_key] = result
                self._cache_timestamps[cache_key] = current_time
        
        # Enregistrer la latence
        latency = (time.perf_counter() - start_time) * 1000
        self.stats_latencies.append(latency)
        
        return result
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _calculate_stats_fast(symbol_id, current_time, window_ms,
                             write_idx, count,
                             timestamps, bids,
                             asks, volumes,
                             buffer_size):
        """
        Calcul rapide des statistiques avec Numba
        """
        # Seuil temporel
        time_threshold = current_time - window_ms
        
        # Arrays temporaires pour les valeurs dans la fenêtre
        valid_bids = np.empty(count, dtype=np.float64)
        valid_asks = np.empty(count, dtype=np.float64)
        valid_volumes = np.empty(count, dtype=np.float64)
        valid_count = 0
        
        # Parcourir le buffer circulaire
        for i in range(count):
            # Index réel dans le buffer
            idx = (write_idx - 1 - i) % buffer_size
            if idx < 0:
                idx += buffer_size
            
            # Vérifier si dans la fenêtre temporelle
            if timestamps[idx] >= time_threshold:
                valid_bids[valid_count] = bids[idx]
                valid_asks[valid_count] = asks[idx]
                valid_volumes[valid_count] = volumes[idx]
                valid_count += 1
            else:
                break  # Les données plus anciennes sont hors fenêtre
        
        # Résultats (17 valeurs)
        results = np.zeros(17, dtype=np.float64)
        
        if valid_count == 0:
            return results
        
        # Nombre d'échantillons
        results[0] = valid_count
        
        # Statistiques bid
        bid_slice = valid_bids[:valid_count]
        results[1] = np.mean(bid_slice)  # mean
        results[2] = np.std(bid_slice)   # std
        results[3] = np.min(bid_slice)   # min
        results[4] = np.max(bid_slice)   # max
        
        # Statistiques ask
        ask_slice = valid_asks[:valid_count]
        results[5] = np.mean(ask_slice)
        results[6] = np.std(ask_slice)
        results[7] = np.min(ask_slice)
        results[8] = np.max(ask_slice)
        
        # Statistiques spread
        spreads = ask_slice - bid_slice
        results[9] = np.mean(spreads)
        results[10] = np.std(spreads)
        results[11] = np.min(spreads)
        results[12] = np.max(spreads)
        
        # Statistiques volume
        vol_slice = valid_volumes[:valid_count]
        results[13] = np.mean(vol_slice)
        results[14] = np.std(vol_slice)
        results[15] = np.sum(vol_slice)  # total
        results[16] = vol_slice[0] if valid_count > 0 else 0  # current
        
        return results
    
    def get_latest(self, symbol: str) -> Optional[MarketSnapshot]:
        """
        Obtient le dernier snapshot - O(1)
        """
        if symbol not in self.symbol_map:
            return None
        
        symbol_id = self.symbol_map[symbol]
        count = self.data_counts[symbol_id]
        
        if count == 0:
            return None
        
        # Index du dernier élément
        write_idx = self.write_indices[symbol_id]
        idx = (write_idx - 1) % self.buffer_size
        if idx < 0:
            idx += self.buffer_size
        
        return MarketSnapshot(
            timestamp=int(self.timestamps[symbol_id, idx]),
            bid=float(self.bids[symbol_id, idx]),
            ask=float(self.asks[symbol_id, idx]),
            last=float(self.lasts[symbol_id, idx]),
            volume=float(self.volumes[symbol_id, idx]),
            bid_size=float(self.bid_sizes[symbol_id, idx]),
            ask_size=float(self.ask_sizes[symbol_id, idx])
        )
    
    def get_microstructure_signals(self, symbol: str, 
                                  lookback: int = 100) -> Optional[Dict[str, float]]:
        """
        Détecte des signaux de microstructure pour le HFT
        """
        if symbol not in self.symbol_map:
            return None
        
        symbol_id = self.symbol_map[symbol]
        count = min(self.data_counts[symbol_id], lookback)
        
        if count < 10:  # Minimum de données nécessaires
            return None
        
        write_idx = self.write_indices[symbol_id]
        
        # Calcul des signaux avec Numba
        signals = self._detect_microstructure_fast(
            self.bids[symbol_id],
            self.asks[symbol_id],
            self.volumes[symbol_id],
            self.bid_sizes[symbol_id],
            self.ask_sizes[symbol_id],
            write_idx,
            count,
            self.buffer_size
        )
        
        return {
            'spread_momentum': float(signals[0]),
            'volume_imbalance': float(signals[1]),
            'bid_ask_ratio': float(signals[2]),
            'micro_trend': float(signals[3]),
            'liquidity_score': float(signals[4])
        }
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _detect_microstructure_fast(bids, asks,
                                   volumes, bid_sizes,
                                   ask_sizes, write_idx,
                                   count, buffer_size):
        """
        Détection rapide de patterns microstructure avec protection contre division par zéro
        """
        results = np.zeros(5, dtype=np.float64)
        
        if count < 2:
            return results
        
        # Arrays pour les calculs
        spreads = np.empty(count, dtype=np.float64)
        mids = np.empty(count, dtype=np.float64)
        
        # Calculer spreads et mid prices
        for i in range(count):
            idx = (write_idx - 1 - i) % buffer_size
            if idx < 0:
                idx += buffer_size
            
            # Protection contre valeurs invalides
            if bids[idx] <= 0 or asks[idx] <= 0:
                spreads[i] = 0.0
                mids[i] = 0.0
            else:
                spreads[i] = asks[idx] - bids[idx]
                mids[i] = (asks[idx] + bids[idx]) / 2
        
        # 1. Spread momentum (variation du spread)
        if count >= 10:
            recent_spread = np.mean(spreads[:5])
            older_spread = np.mean(spreads[5:10])
            if older_spread > 0.0001:  # Éviter division par zéro
                results[0] = (recent_spread - older_spread) / older_spread
            else:
                results[0] = 0.0
        
        # 2. Volume imbalance
        total_bid_size = 0.0
        total_ask_size = 0.0
        for i in range(min(count, 20)):
            idx = (write_idx - 1 - i) % buffer_size
            if idx < 0:
                idx += buffer_size
            # Correction : bid_sizes[idx] et ask_sizes[idx] sont des floats
            total_bid_size += max(0.0, float(bid_sizes[idx]))
            total_ask_size += max(0.0, float(ask_sizes[idx]))
        
        total_volume = total_bid_size + total_ask_size
        if total_volume > 0.0001:
            results[1] = (total_bid_size - total_ask_size) / total_volume
        else:
            results[1] = 0.0
        
        # 3. Bid-Ask pressure ratio
        if total_ask_size > 0.0001:
            results[2] = total_bid_size / total_ask_size
        else:
            results[2] = 1.0  # Neutral si pas de données
        
        # 4. Micro trend (pente des mid prices)
        if count >= 5:
            # Régression linéaire simple
            x = np.arange(5, dtype=np.float64)
            y = mids[:5]
            
            # Vérifier que nous avons des données valides
            valid_count = 0
            for i in range(5):
                if y[i] > 0:
                    valid_count += 1
            
            if valid_count >= 3:  # Au moins 3 points valides
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                
                if y_mean > 0.0001:  # Protection contre division par zéro
                    num = 0.0
                    den = 0.0
                    for i in range(5):
                        if y[i] > 0:  # Ignorer les valeurs invalides
                            num += (x[i] - x_mean) * (y[i] - y_mean)
                            den += (x[i] - x_mean) ** 2
                    
                    if den > 0.0001:
                        slope = num / den
                        # Normaliser par le prix moyen
                        results[3] = slope / y_mean * 1000  # En basis points
                    else:
                        results[3] = 0.0
                else:
                    results[3] = 0.0
            else:
                results[3] = 0.0
        
        # 5. Liquidity score (inverse du spread moyen normalisé)
        valid_spreads = 0
        sum_spreads = 0.0
        sum_mids = 0.0
        
        for i in range(min(count, 20)):
            if spreads[i] > 0 and mids[i] > 0:
                sum_spreads += spreads[i]
                sum_mids += mids[i]
                valid_spreads += 1
        
        if valid_spreads > 0 and sum_mids > 0.0001:
            avg_spread = sum_spreads / valid_spreads
            avg_mid = sum_mids / valid_spreads
            spread_bps = (avg_spread / avg_mid) * 10000
            results[4] = 1.0 / (1.0 + spread_bps / 10.0)  # Score 0-1
        else:
            results[4] = 0.5  # Score neutre si pas de données
        
        return results
    
    def get_performance_stats(self) -> dict:
        """Retourne les statistiques de performance du buffer"""
        update_latencies = list(self.update_latencies)
        stats_latencies = list(self.stats_latencies)
        
        return {
            'update_latency_mean_ms': float(np.mean(update_latencies)) if update_latencies else 0.0,
            'update_latency_p95_ms': float(np.percentile(update_latencies, 95)) if update_latencies else 0.0,
            'update_latency_p99_ms': float(np.percentile(update_latencies, 99)) if update_latencies else 0.0,
            'stats_latency_mean_ms': float(np.mean(stats_latencies)) if stats_latencies else 0.0,
            'stats_latency_p95_ms': float(np.percentile(stats_latencies, 95)) if stats_latencies else 0.0,
            'symbols_count': int(len(self.symbol_map)),
            'total_updates': int(sum(self.data_counts)),
            'cache_size': int(len(self._cache))
        }
    
    def clear_symbol(self, symbol: str):
        """Efface les données d'un symbole"""
        if symbol not in self.symbol_map:
            return
        
        with self.lock:
            symbol_id = self.symbol_map[symbol]
            self.write_indices[symbol_id] = 0
            self.data_counts[symbol_id] = 0
            
            # Effacer le cache
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{symbol}:")]
            for k in keys_to_remove:
                del self._cache[k]
                del self._cache_timestamps[k]


# Test rapide si exécuté directement
if __name__ == "__main__":
    import time
    
    print("Test du FastMarketBuffer...")
    buffer = FastMarketBuffer()
    
    # Test de performance - updates
    print("\n1. Test updates (objectif < 0.1ms):")
    start = time.perf_counter()
    for i in range(10000):
        buffer.update(
            'BTC/USDT',
            bid=50000 + i * 0.1,
            ask=50001 + i * 0.1,
            last=50000.5 + i * 0.1,
            volume=100 + i
        )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"   10,000 updates en {elapsed:.1f}ms ({elapsed/10000:.3f}ms par update)")
    
    # Test de performance - stats
    print("\n2. Test calcul stats (objectif < 1ms):")
    start = time.perf_counter()
    for _ in range(100):
        stats = buffer.get_stats('BTC/USDT', window_ms=1000)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"   100 calculs stats en {elapsed:.1f}ms ({elapsed/100:.3f}ms par calcul)")
    
    # Afficher les stats
    if stats and isinstance(stats, dict):
        print(f"\n3. Exemple de stats:")
        print(f"   Samples: {stats['samples'] if 'samples' in stats else 0}")
        bid = stats['bid'] if isinstance(stats.get('bid'), dict) else None
        spread = stats['spread'] if isinstance(stats.get('spread'), dict) else None
        volume = stats['volume'] if isinstance(stats.get('volume'), dict) else None
        bid_mean = bid.get('mean', 0.0) if isinstance(bid, dict) else 0.0
        spread_mean = spread.get('mean', 0.0) if isinstance(spread, dict) else 0.0
        volume_total = volume.get('total', 0.0) if isinstance(volume, dict) else 0.0
        print(f"   Bid mean: ${bid_mean:.2f}")
        print(f"   Spread mean: ${spread_mean:.4f}")
        print(f"   Volume total: {volume_total:.0f}")
    
    # Test microstructure
    micro = buffer.get_microstructure_signals('BTC/USDT')
    if micro:
        print(f"\n4. Signaux microstructure:")
        for signal, value in micro.items():
            print(f"   {signal}: {value:.4f}")
    
    # Performance globale
    perf = buffer.get_performance_stats()
    print(f"\n5. Performance du buffer:")
    print(f"   Update latency: {perf['update_latency_mean_ms']:.3f}ms (mean)")
    print(f"   Stats latency: {perf['stats_latency_mean_ms']:.3f}ms (mean)")
    print(f"   Total updates: {perf['total_updates']:,}")