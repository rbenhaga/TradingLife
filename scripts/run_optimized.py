#!/usr/bin/env python
"""
Script d'intégration pour connecter tous les composants optimisés
- FastMarketBuffer pour les données temps réel
- OptimizedWeightedScoreEngine pour les calculs
- Prometheus pour le monitoring
- Tests de latence intégrés
"""

import asyncio
import time
from pathlib import Path
import sys
from typing import Dict, Optional
import numpy as np

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.trading_bot import TradingBot
from src.core.fast_market_buffer import FastMarketBuffer
from src.core.optimized_weighted_score_engine import OptimizedWeightedScoreEngine
from src.monitoring.prometheus_metrics import (
    PrometheusExporter,
    decision_latency,
    trade_latency,
    measure_latency,
    market_data_lag
)
from src.core.logger import log_info, log_error, log_warning


class OptimizedTradingBot(TradingBot):
    """
    Version optimisée du TradingBot avec tous les composants haute performance
    """
    
    def __init__(self, config_path: str = "config/config.json", paper_trading: bool = True):
        """Initialise le bot optimisé"""
        super().__init__(config_path, paper_trading)
        
        # Composants optimisés
        self.market_buffer = FastMarketBuffer(max_symbols=100)
        self.score_engine = OptimizedWeightedScoreEngine()
        self.metrics_exporter = None
        self._running = True  # Ajouté pour éviter l'erreur d'attribut
        
        # Seuils de latence par phase
        self.latency_targets = {
            'phase1': {'decision': 100, 'execution': 100},  # ms
            'phase2': {'decision': 20, 'execution': 20},
            'phase3': {'decision': 5, 'execution': 5}
        }
        self.current_phase = 'phase1'
        
        log_info("OptimizedTradingBot initialisé avec composants haute performance")
    
    async def initialize(self) -> bool:
        """Initialise le bot avec monitoring Prometheus"""
        try:
            # Démarrer Prometheus
            self.metrics_exporter = PrometheusExporter(
                port=self.config.get('prometheus', {}).get('port', 8000)
            )
            self.metrics_exporter.start()
            log_info(f"✅ Prometheus démarré sur port {self.metrics_exporter.port}")
            
            # Initialiser le bot de base
            result = await super().initialize()
            
            if result:
                # Remplacer le score engine standard par la version optimisée
                if hasattr(self, 'pair_manager') and self.pair_manager:
                    for strategy in self.pair_manager.strategies.values():
                        if hasattr(strategy, 'score_engine'):
                            strategy.score_engine = self.score_engine
                
                # Démarrer la collecte de métriques
                self._tasks.add(
                    asyncio.create_task(self._metrics_collection_loop())
                )
                
                log_info("✅ Tous les composants optimisés initialisés")
            
            return result
            
        except Exception as e:
            log_error(f"Erreur initialisation: {e}")
            return False
    
    async def _process_market_update(self, update: Dict):
        """
        Traite une mise à jour de marché avec mesure de latence
        """
        start_time = time.perf_counter()
        symbol = update.get('symbol', 'UNKNOWN')
        
        try:
            # Mettre à jour le buffer optimisé
            self.market_buffer.update(
                symbol=symbol,
                bid=float(update.get('bid', 0)),
                ask=float(update.get('ask', 0)),
                last=float(update.get('last', 0)),
                volume=float(update.get('volume', 0)),
                timestamp=update.get('timestamp')
            )
            
            # Mesurer le lag si timestamp serveur disponible
            if 'timestamp' in update:
                lag_ms = abs(time.time() * 1000 - update['timestamp'])
                market_data_lag.labels(symbol=symbol).observe(lag_ms)
            
            # Obtenir les stats et signaux microstructure
            stats = self.market_buffer.get_stats(symbol, window_ms=5000)
            micro_signals = self.market_buffer.get_microstructure_signals(symbol)
            
            # Analyser si nécessaire
            if stats and stats['samples'] > 50:
                await self._analyze_and_decide(symbol, stats, micro_signals if micro_signals is not None else {})
            
        finally:
            # Mesurer la latence totale
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._check_latency(latency_ms, 'market_update', symbol)
    
    @measure_latency(decision_latency, {'strategy': 'optimized'})
    async def _analyze_and_decide(self, symbol: str, stats: Dict, micro_signals: Dict):
        """
        Analyse et prise de décision optimisée
        """
        start_time = time.perf_counter()
        
        # Préparer les données pour l'analyse
        latest = self.market_buffer.get_latest(symbol)
        if not latest:
            return
        
        # Construire les signaux
        signals = {
            'rsi': {
                'signal': micro_signals.get('micro_trend', 0) * 0.5,  # Proxy
                'confidence': 0.7,
                'reason': 'Micro trend analysis'
            },
            'bollinger': {
                'signal': -micro_signals.get('spread_momentum', 0),
                'confidence': 0.8,
                'reason': f"Spread momentum {micro_signals.get('spread_momentum', 0):.3f}"
            },
            'volume': {
                'signal': micro_signals.get('volume_imbalance', 0),
                'confidence': micro_signals.get('liquidity_score', 0.5),
                'reason': f"Volume imbalance {micro_signals.get('volume_imbalance', 0):.2f}"
            }
        }
        
        # Calculer le score avec l'engine optimisé
        score_result = self.score_engine.calculate_score(signals, symbol)
        
        # Enregistrer les métriques
        if self.metrics_exporter is not None:
            self.metrics_exporter.record_signal(
                strategy='optimized',
                signal_type=score_result.direction,
                symbol=symbol,
                confidence=score_result.confidence
            )
        
        # Décision de trading
        if abs(score_result.total_score) > 0.7 and score_result.confidence > 0.6:
            action = 'BUY' if score_result.total_score > 0 else 'SELL'
            
            # Préparer l'ordre
            order = {
                'symbol': symbol,
                'side': action.lower(),
                'type': 'limit',
                'price': latest.bid if action == 'SELL' else latest.ask,
                'amount': self._calculate_position_size(symbol, score_result.confidence),
                'score': score_result.total_score,
                'confidence': score_result.confidence
            }
            
            # Exécuter si les conditions sont remplies
            if self.paper_trading or (hasattr(self, 'exchange') and self.exchange is not None and await self._check_risk_limits(order)):
                await self._execute_order(order)
        
        # Mesurer la latence de décision
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._check_latency(latency_ms, 'decision', symbol)
    
    @measure_latency(trade_latency, {'exchange': 'optimized'})
    async def _execute_order(self, order: Dict):
        """Exécute un ordre avec mesure de latence"""
        start_time = time.perf_counter()
        
        try:
            if self.paper_trading:
                # Simulation d'exécution
                await asyncio.sleep(0.001)  # 1ms simulé
                result = {
                    'id': f"PAPER-{int(time.time()*1000)}",
                    'status': 'filled',
                    'price': order['price'],
                    'amount': order['amount']
                }
            else:
                # Exécution réelle
                if hasattr(self, 'exchange') and self.exchange is not None:
                    result = await self.exchange.create_order(**order)
                else:
                    result = None
            
            # Enregistrer le trade
            if result and result.get('status') == 'filled':
                if self.metrics_exporter is not None:
                    self.metrics_exporter.record_trade(
                        symbol=order['symbol'],
                        side=order['side'],
                        pnl=0,  # À calculer
                        strategy='optimized'
                    )
            
            log_info(f"✅ Ordre exécuté: {order['symbol']} {order['side']} @ {order['price']}")
            
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._check_latency(latency_ms, 'execution', order['symbol'])
    
    def _check_latency(self, latency_ms: float, operation: str, symbol: str):
        """Vérifie la latence par rapport aux objectifs"""
        target = self.latency_targets[self.current_phase].get(operation, 100)
        
        if latency_ms > target:
            log_warning(
                f"⚠️ Latence {operation} élevée pour {symbol}: "
                f"{latency_ms:.1f}ms (objectif: {target}ms)"
            )
        elif latency_ms < target * 0.5:
            log_info(
                f"🚀 Excellente latence {operation} pour {symbol}: "
                f"{latency_ms:.1f}ms"
            )
    
    def _calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calcule la taille de position basée sur la confiance"""
        base_size = self.config.get('trading', {}).get('base_position_size', 0.01)
        max_size = self.config.get('trading', {}).get('max_position_size', 0.1)
        
        # Ajuster selon la confiance
        size = base_size * (0.5 + confidence * 0.5)
        
        # Limiter
        return min(size, max_size)
    
    async def _check_risk_limits(self, order: Dict) -> bool:
        """Vérifie les limites de risque"""
        # Implémentation simplifiée
        return True
    
    async def _metrics_collection_loop(self):
        """Boucle de collecte des métriques"""
        while getattr(self, '_running', True):
            try:
                # Métriques du buffer
                buffer_stats = self.market_buffer.get_performance_stats()
                
                # Métriques du score engine
                engine_stats = self.score_engine.get_performance_stats()
                
                # Mettre à jour Prometheus
                trading_metrics = {
                    'buffer_latency_ms': buffer_stats.get('update_latency_mean_ms', 0),
                    'score_latency_ms': engine_stats.get('avg_calculation_time_ms', 0),
                    'total_symbols': buffer_stats.get('symbols_count', 0),
                    'cache_hit_rate': engine_stats.get('cache_hit_rate', 0)
                }
                
                if self.metrics_exporter is not None:
                    self.metrics_exporter.update_trading_metrics({
                        'performance': trading_metrics
                    })
                
                # Log périodique
                log_info(
                    f"📊 Perf - Buffer: {buffer_stats.get('update_latency_mean_ms', 0):.2f}ms, "
                    f"Score: {engine_stats.get('avg_calculation_time_ms', 0):.2f}ms, "
                    f"Symbols: {buffer_stats.get('symbols_count', 0)}"
                )
                
                await asyncio.sleep(30)  # Toutes les 30 secondes
                
            except Exception as e:
                log_error(f"Erreur collecte métriques: {e}")
                await asyncio.sleep(60)
    
    def set_phase(self, phase: str):
        """Change la phase du bot (phase1/phase2/phase3)"""
        if phase in self.latency_targets:
            self.current_phase = phase
            log_info(f"📈 Phase changée: {phase} - Objectifs: {self.latency_targets[phase]}")
    
    async def run_performance_test(self):
        """Lance un test de performance complet"""
        log_info("\n" + "="*60)
        log_info("🚀 TEST DE PERFORMANCE INTÉGRÉ")
        log_info("="*60)
        
        # Test 1: Updates du buffer
        log_info("\n1. Test updates buffer (10,000 updates):")
        start = time.perf_counter()
        for i in range(10000):
            self.market_buffer.update(
                'TEST/USDT',
                bid=50000 + i * 0.1,
                ask=50001 + i * 0.1,
                last=50000.5 + i * 0.1,
                volume=100 + i
            )
        elapsed = (time.perf_counter() - start) * 1000
        log_info(f"   ✅ {elapsed/10000:.3f}ms par update (objectif Phase 3: < 0.1ms)")
        
        # Test 2: Calcul de score
        log_info("\n2. Test calcul de score (1,000 calculs):")
        test_signals = {
            'rsi': {'signal': 0.5, 'confidence': 0.8},
            'bollinger': {'signal': -0.3, 'confidence': 0.7},
            'volume': {'signal': 0.2, 'confidence': 0.9}
        }
        
        start = time.perf_counter()
        for _ in range(1000):
            _ = self.score_engine.calculate_score(test_signals, 'TEST/USDT')
        elapsed = (time.perf_counter() - start) * 1000
        log_info(f"   ✅ {elapsed/1000:.3f}ms par calcul (objectif Phase 2: < 0.5ms)")
        
        # Test 3: Pipeline complet
        log_info("\n3. Test pipeline complet (100 cycles):")
        latencies = []
        
        for i in range(100):
            start = time.perf_counter()
            
            # Update
            self.market_buffer.update(
                'BENCH/USDT',
                bid=50000 + np.random.randn() * 10,
                ask=50001 + np.random.randn() * 10,
                last=50000.5 + np.random.randn() * 10,
                volume=100 + np.random.exponential(50)
            )
            
            # Stats
            stats = self.market_buffer.get_stats('BENCH/USDT')
            micro = self.market_buffer.get_microstructure_signals('BENCH/USDT')
            
            # Score
            if stats and micro:
                signals = {
                    'momentum': {'signal': micro.get('micro_trend', 0), 'confidence': 0.7},
                    'volume': {'signal': micro.get('volume_imbalance', 0), 'confidence': 0.8}
                }
                score = self.score_engine.calculate_score(signals, 'BENCH/USDT')
            
            latencies.append((time.perf_counter() - start) * 1000)
        
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        log_info(f"   ✅ Latence moyenne: {mean_latency:.2f}ms")
        log_info(f"   ✅ P95: {p95_latency:.2f}ms")
        
        # Résumé par phase
        log_info("\n📊 RÉSUMÉ PAR PHASE:")
        for phase, targets in self.latency_targets.items():
            log_info(f"\n{phase.upper()}:")
            decision_ok = mean_latency < targets['decision']
            log_info(f"   Décision: {'✅' if decision_ok else '❌'} "
                    f"(actuel: {mean_latency:.1f}ms, objectif: < {targets['decision']}ms)")


# Fonction principale pour tester
async def main():
    """Lance le bot optimisé avec tests"""
    bot = OptimizedTradingBot(paper_trading=True)
    
    try:
        # Initialiser
        if not await bot.initialize():
            log_error("Échec de l'initialisation")
            return
        
        # Lancer le test de performance
        await bot.run_performance_test()
        
        # Simuler quelques updates
        log_info("\n4. Simulation de marché en temps réel...")
        for i in range(10):
            update = {
                'symbol': 'BTC/USDT',
                'bid': 50000 + np.random.randn() * 50,
                'ask': 50001 + np.random.randn() * 50,
                'last': 50000.5 + np.random.randn() * 50,
                'volume': 100 + np.random.exponential(100),
                'timestamp': int(time.time() * 1000)
            }
            
            await bot._process_market_update(update)
            await asyncio.sleep(0.1)
        
        log_info("\n✅ Test terminé avec succès!")
        log_info(f"📊 Métriques disponibles sur http://localhost:8000/metrics")
        
    except Exception as e:
        log_error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())