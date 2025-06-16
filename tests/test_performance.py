# tests/test_performance.py
"""
Tests de performance pour vérifier les objectifs de latence
Utilise pytest-benchmark pour des mesures précises
"""

import pytest
import numpy as np
import time
import asyncio
from unittest.mock import Mock, AsyncMock

from src.core.fast_market_buffer import FastMarketBuffer
from src.core.weighted_score_engine import WeightedScoreEngine
from src.core.websocket_market_feed import WebSocketMarketFeed
from src.strategies.scalping_strategy import ScalpingStrategy


class TestPerformance:
    """Suite de tests de performance pour les composants critiques"""
    
    @pytest.fixture
    def market_buffer(self):
        """Fixture pour le buffer de marché"""
        return FastMarketBuffer(max_symbols=100, buffer_size=5000)
    
    @pytest.fixture
    def sample_data(self):
        """Génère des données de test réalistes"""
        return {
            'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT'],
            'prices': np.random.uniform(0.1, 50000, 1000),
            'volumes': np.random.uniform(100, 10000, 1000)
        }
    
    def test_market_buffer_update_latency(self, benchmark, market_buffer):
        """Test: Mise à jour du buffer < 0.1ms"""
        
        def update_single():
            market_buffer.update(
                'BTC/USDT',
                bid=50000.0,
                ask=50001.0,
                last=50000.5,
                volume=123.45,
                bid_size=10.0,
                ask_size=8.5
            )
        
        # Benchmark avec au moins 1000 rounds
        result = benchmark.pedantic(update_single, rounds=1000, iterations=10)
        
        # Vérifier que la latence médiane est < 0.1ms (100 microsecondes)
        assert benchmark.stats['median'] < 0.0001
        print(f"\n✅ Latence update buffer: {benchmark.stats['median']*1000:.3f}ms")
    
    def test_market_buffer_batch_update(self, benchmark, market_buffer, sample_data):
        """Test: Mise à jour batch de 50 symboles"""
        
        def update_batch():
            for i in range(50):
                symbol = f"COIN{i}/USDT"
                market_buffer.update(
                    symbol,
                    bid=float(sample_data['prices'][i]),
                    ask=float(sample_data['prices'][i] * 1.001),
                    last=float(sample_data['prices'][i] * 1.0005),
                    volume=float(sample_data['volumes'][i])
                )
        
        result = benchmark(update_batch)
        
        # 50 updates en < 5ms
        assert benchmark.stats['median'] < 0.005
        print(f"\n✅ Latence batch 50 symboles: {benchmark.stats['median']*1000:.3f}ms")
    
    def test_stats_calculation_performance(self, benchmark, market_buffer):
        """Test: Calcul des statistiques < 1ms"""
        
        # Pré-remplir le buffer
        for i in range(1000):
            market_buffer.update(
                'BTC/USDT',
                bid=50000 + np.random.randn() * 10,
                ask=50001 + np.random.randn() * 10,
                last=50000.5 + np.random.randn() * 10,
                volume=100 + np.random.randn() * 10
            )
        
        def calculate_stats():
            return market_buffer.get_stats('BTC/USDT', window_ms=1000)
        
        result = benchmark(calculate_stats)
        
        # Stats en < 1ms
        assert benchmark.stats['median'] < 0.001
        print(f"\n✅ Latence calcul stats: {benchmark.stats['median']*1000:.3f}ms")
    
    def test_microstructure_detection(self, benchmark, market_buffer):
        """Test: Détection patterns microstructure < 2ms"""
        
        # Simuler un carnet d'ordres actif
        for i in range(200):
            spread = 0.5 + 0.1 * np.sin(i / 10)  # Spread oscillant
            market_buffer.update(
                'BTC/USDT',
                bid=50000 - spread/2,
                ask=50000 + spread/2,
                last=50000,
                volume=100 * (1 + 0.5 * np.sin(i / 5))
            )
        
        def detect_patterns():
            return market_buffer.get_microstructure_signals('BTC/USDT')
        
        result = benchmark(detect_patterns)
        
        # Détection en < 2ms
        assert benchmark.stats['median'] < 0.002
        print(f"\n✅ Latence détection microstructure: {benchmark.stats['median']*1000:.3f}ms")
    
    def test_strategy_decision_latency(self, benchmark):
        """Test: Décision de trading complète < 20ms (objectif Phase 2)"""
        
        # Mock des composants
        mock_buffer = FastMarketBuffer()
        mock_engine = WeightedScoreEngine()
        
        # Pré-remplir avec des données
        for _ in range(100):
            mock_buffer.update(
                'BTC/USDT',
                bid=50000, ask=50001, last=50000.5,
                volume=100
            )
        
        strategy = ScalpingStrategy('BTC/USDT')
        
        def make_decision():
            # Récupérer les données
            snapshot = mock_buffer.get_latest('BTC/USDT')
            stats = mock_buffer.get_stats('BTC/USDT')
            micro = mock_buffer.get_microstructure_signals('BTC/USDT')
            
            # Générer le signal
            signal = strategy.generate_signal(snapshot, stats, micro)
            
            return signal
        
        result = benchmark(make_decision)
        
        # Décision complète en < 20ms
        assert benchmark.stats['median'] < 0.020
        print(f"\n✅ Latence décision complète: {benchmark.stats['median']*1000:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_websocket_processing_latency(self, benchmark):
        """Test: Traitement message WebSocket < 5ms"""
        
        buffer = FastMarketBuffer()
        
        # Message WebSocket simulé
        ws_message = {
            'e': '24hrTicker',
            's': 'BTCUSDT',
            'b': '50000.00',
            'a': '50001.00',
            'c': '50000.50',
            'v': '1234.56'
        }
        
        async def process_message():
            # Parser et mettre à jour le buffer
            symbol = ws_message['s']
            buffer.update(
                symbol,
                bid=float(ws_message['b']),
                ask=float(ws_message['a']),
                last=float(ws_message['c']),
                volume=float(ws_message['v'])
            )
            
            # Obtenir les signaux
            return buffer.get_microstructure_signals(symbol)
        
        # Benchmark async
        result = await benchmark(process_message)
        
        # Traitement en < 5ms
        assert benchmark.stats['median'] < 0.005
        print(f"\n✅ Latence traitement WebSocket: {benchmark.stats['median']*1000:.3f}ms")
    
    def test_memory_efficiency(self, market_buffer):
        """Test: Utilisation mémoire efficace"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Mémoire initiale
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Remplir le buffer avec 100 symboles
        for i in range(100):
            symbol = f"COIN{i}/USDT"
            for j in range(1000):
                market_buffer.update(
                    symbol,
                    bid=1000 + j,
                    ask=1001 + j,
                    last=1000.5 + j,
                    volume=100
                )
        
        # Mémoire après
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        # Vérifier que l'utilisation est raisonnable (< 100MB pour 100k entrées)
        assert mem_used < 100
        print(f"\n✅ Mémoire utilisée: {mem_used:.1f}MB pour 100k entrées")
    
    def test_concurrent_access(self, benchmark, market_buffer):
        """Test: Accès concurrent thread-safe"""
        import threading
        import concurrent.futures
        
        def update_worker(thread_id):
            for i in range(100):
                market_buffer.update(
                    f"THREAD{thread_id}/USDT",
                    bid=1000 + i,
                    ask=1001 + i,
                    last=1000.5 + i,
                    volume=100
                )
        
        def concurrent_updates():
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(update_worker, i) for i in range(10)]
                concurrent.futures.wait(futures)
        
        result = benchmark(concurrent_updates)
        
        # Vérifier que tous les symboles sont présents
        assert len(market_buffer.symbol_map) == 10
        print(f"\n✅ Latence 10 threads concurrent: {benchmark.stats['median']*1000:.3f}ms")


# Configuration pytest-benchmark
pytest_plugins = ['pytest_benchmark']

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark"
    )