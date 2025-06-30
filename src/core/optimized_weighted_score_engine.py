# src/core/optimized_weighted_score_engine.py
"""
Moteur de score pondéré optimisé avec Numba pour haute performance
Objectif: < 0.5ms pour calcul de score complet
"""

import pandas as pd
import numpy as np
from numba import jit, njit, float64, int32
from numba.typed import Dict as NumbaDict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

from .weighted_score_engine import TradingScore, SignalComponent

logger = logging.getLogger('OptimizedWeightedScoreEngine')


class OptimizedWeightedScoreEngine:
    """
    Version optimisée du moteur de score avec calculs Numba
    Compatible avec l'interface existante
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialise le moteur de score optimisé
        
        Args:
            weights: Poids personnalisés pour chaque indicateur
        """
        # Poids par défaut
        self.default_weights = {
            'rsi': 0.20,
            'bollinger': 0.20,
            'macd': 0.15,
            'volume': 0.15,
            'ma_cross': 0.10,
            'momentum': 0.10,
            'volatility': 0.10
        }
        
        # Utiliser les poids personnalisés ou par défaut
        self.weights = weights or self.default_weights.copy()
        self._normalize_weights()
        
        # Convertir en arrays numpy pour Numba
        self.indicator_names = list(self.weights.keys())
        self.weights_array = np.array([self.weights[ind] for ind in self.indicator_names], dtype=np.float64)
        
        # Pré-compiler les fonctions Numba
        self._precompile_numba_functions()
        
        # Cache pour les calculs
        self._cache = {}
        self._cache_ttl = 100  # ms
        
        logger.info(f"OptimizedScoreEngine initialisé avec {len(self.weights)} indicateurs")
    
    def _normalize_weights(self):
        """Normalise les poids pour qu'ils somment à 1"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def _precompile_numba_functions(self):
        """Pré-compile les fonctions Numba pour éviter la latence au premier appel"""
        # Données factices pour la compilation
        dummy_values = np.array([0.5, -0.3, 0.2, 0.0, 0.1, -0.2, 0.3], dtype=np.float64)
        dummy_weights = np.array([0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1], dtype=np.float64)
        dummy_confidences = np.array([0.8, 0.7, 0.9, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        
        # Forcer la compilation
        _ = self._calculate_score_fast(dummy_values, dummy_weights, dummy_confidences)
        _ = self._calculate_rsi_signal_fast(np.array([50.0]), 30.0, 70.0)
        _ = self._calculate_bollinger_signal_fast(50000.0, 51000.0, 49000.0)
        _ = self._calculate_macd_signal_fast(0.5, 0.3)
        _ = self._calculate_volume_signal_fast(1.5, 1.2)
    
    @staticmethod
    @njit(cache=True)
    def _calculate_score_fast(signal_values, weights, confidences):
        """
        Calcul rapide du score pondéré avec Numba
        
        Returns:
            (total_score, average_confidence)
        """
        total_score = 0.0
        total_confidence = 0.0
        weights_sum = 0.0
        
        for i in range(len(signal_values)):
            if not np.isnan(signal_values[i]):
                # Limiter la valeur du signal entre -1 et 1
                signal = max(-1.0, min(1.0, signal_values[i]))
                conf = max(0.0, min(1.0, confidences[i]))
                
                # Contribution pondérée
                total_score += signal * weights[i]
                total_confidence += conf * weights[i]
                weights_sum += weights[i]
        
        # Normaliser
        if weights_sum > 0:
            total_confidence = total_confidence / weights_sum
        else:
            total_confidence = 0.0
        
        # Limiter le score final
        total_score = max(-1.0, min(1.0, total_score))
        
        return total_score, total_confidence
    
    @staticmethod
    @njit(cache=True)
    def _calculate_rsi_signal_fast(rsi_values, oversold, overbought):
        """Calcul rapide du signal RSI"""
        if len(rsi_values) == 0:
            return 0.0, 0.0
        
        rsi = rsi_values[-1]
        
        if np.isnan(rsi):
            return 0.0, 0.0
        
        # Signal et confiance
        if rsi < oversold:
            signal = (oversold - rsi) / oversold  # Plus c'est bas, plus c'est fort
            confidence = min(1.0, (oversold - rsi) / 10.0)
        elif rsi > overbought:
            signal = -(rsi - overbought) / (100 - overbought)
            confidence = min(1.0, (rsi - overbought) / 10.0)
        else:
            # Zone neutre
            signal = (50 - rsi) / 50 * 0.3  # Signal faible
            confidence = 0.3
        
        return signal, confidence
    
    @staticmethod
    @njit(cache=True)
    def _calculate_bollinger_signal_fast(price, upper, lower):
        """Calcul rapide du signal Bollinger Bands"""
        if upper <= lower or np.isnan(price) or np.isnan(upper) or np.isnan(lower):
            return 0.0, 0.0
        
        # Position relative dans les bandes
        band_width = upper - lower
        position = (price - lower) / band_width if band_width > 0 else 0.5
        
        # Signal basé sur la position
        if position < 0.2:  # Proche de la bande inférieure
            signal = (0.2 - position) * 5  # Max 1.0
            confidence = 0.8
        elif position > 0.8:  # Proche de la bande supérieure
            signal = -(position - 0.8) * 5  # Max -1.0
            confidence = 0.8
        else:
            signal = 0.0
            confidence = 0.2
        
        return max(-1.0, min(1.0, signal)), confidence
    
    @staticmethod
    @njit(cache=True)
    def _calculate_macd_signal_fast(macd, signal_line):
        """Calcul rapide du signal MACD"""
        if np.isnan(macd) or np.isnan(signal_line):
            return 0.0, 0.0
        
        # Divergence MACD
        divergence = macd - signal_line
        
        # Normaliser le signal (basé sur l'observation empirique)
        normalized_signal = np.tanh(divergence * 10)  # Sensibilité ajustable
        
        # Confiance basée sur la force du signal
        confidence = min(1.0, abs(divergence) * 50)
        
        return normalized_signal, confidence
    
    @staticmethod
    @njit(cache=True)
    def _calculate_volume_signal_fast(current_ratio, threshold):
        """Calcul rapide du signal de volume"""
        if np.isnan(current_ratio) or current_ratio <= 0:
            return 0.0, 0.0
        
        # Signal basé sur le ratio de volume
        if current_ratio > threshold:
            signal = min(1.0, (current_ratio - threshold) / threshold)
            confidence = min(0.9, signal)
        else:
            signal = 0.0
            confidence = 0.1
        
        return signal, confidence
    
    def calculate_score(self, signals: Dict[str, Dict], symbol: str = "UNKNOWN") -> TradingScore:
        """
        Calcule le score pondéré optimisé
        
        Args:
            signals: Dict des signaux par indicateur
            symbol: Symbole tradé
        
        Returns:
            TradingScore avec tous les détails
        """
        start_time = time.perf_counter()
        
        # Vérifier le cache
        cache_key = f"{symbol}:{hash(str(signals))}"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (time.time() * 1000 - cached_time) < self._cache_ttl:
                return cached_result
        
        # Préparer les arrays pour Numba
        n_indicators = len(self.indicator_names)
        signal_values = np.full(n_indicators, np.nan, dtype=np.float64)
        confidences = np.full(n_indicators, 0.5, dtype=np.float64)
        reasons = []
        
        # Extraire les valeurs
        for i, indicator in enumerate(self.indicator_names):
            if indicator in signals:
                signal_data = signals[indicator]
                
                if isinstance(signal_data, dict):
                    signal_values[i] = float(signal_data.get('signal', 0))
                    confidences[i] = float(signal_data.get('confidence', 0.5))
                    reasons.append(signal_data.get('reason', f'{indicator} signal'))
                else:
                    signal_values[i] = float(signal_data)
                    reasons.append(f'{indicator} signal')
            else:
                reasons.append(f'{indicator} missing')
        
        # Calcul optimisé avec Numba
        total_score, avg_confidence = self._calculate_score_fast(
            signal_values, self.weights_array, confidences
        )
        
        # Créer les composants de signal
        signal_components = []
        for i, indicator in enumerate(self.indicator_names):
            if not np.isnan(signal_values[i]):
                component = SignalComponent(
                    name=indicator,
                    value=float(signal_values[i]),
                    weight=float(self.weights_array[i]),
                    weighted_value=float(signal_values[i] * self.weights_array[i]),
                    confidence=float(confidences[i]),
                    reason=reasons[i]
                )
                signal_components.append(component)
        
        # Créer le TradingScore
        trading_score = TradingScore(
            symbol=symbol,
            timestamp=datetime.now(),
            total_score=float(total_score),
            confidence=float(avg_confidence),
            direction='',
            signals=signal_components,
            metadata={
                'calculation_time_ms': (time.perf_counter() - start_time) * 1000,
                'indicators_used': len(signal_components),
                'cache_hit': False
            }
        )
        
        trading_score.direction = trading_score.get_action()
        
        # Mettre en cache
        self._cache[cache_key] = (time.time() * 1000, trading_score)
        
        return trading_score
    
    def analyze_indicators_fast(self, data: Dict[str, np.ndarray], 
                               symbol: str = "UNKNOWN") -> Dict[str, Dict]:
        """
        Analyse rapide des indicateurs avec arrays numpy
        
        Args:
            data: Dict avec arrays numpy pour chaque indicateur
                 Ex: {'close': np.array(...), 'rsi': np.array(...), ...}
            symbol: Symbole tradé
        
        Returns:
            Dict des signaux par indicateur
        """
        signals = {}
        
        # RSI
        if 'rsi' in data and len(data['rsi']) > 0:
            rsi_signal, rsi_conf = self._calculate_rsi_signal_fast(
                data['rsi'], 30.0, 70.0
            )
            signals['rsi'] = {
                'signal': float(rsi_signal),
                'confidence': float(rsi_conf),
                'reason': f"RSI {data['rsi'][-1]:.1f}"
            }
        
        # Bollinger Bands
        if all(k in data for k in ['close', 'bb_upper', 'bb_lower']):
            if len(data['close']) > 0:
                bb_signal, bb_conf = self._calculate_bollinger_signal_fast(
                    data['close'][-1],
                    data['bb_upper'][-1],
                    data['bb_lower'][-1]
                )
                signals['bollinger'] = {
                    'signal': float(bb_signal),
                    'confidence': float(bb_conf),
                    'reason': f"Price at {((data['close'][-1] - data['bb_lower'][-1]) / (data['bb_upper'][-1] - data['bb_lower'][-1]) * 100):.0f}% of bands"
                }
        
        # MACD
        if 'macd' in data and 'macd_signal' in data:
            if len(data['macd']) > 0:
                macd_signal, macd_conf = self._calculate_macd_signal_fast(
                    data['macd'][-1],
                    data['macd_signal'][-1]
                )
                signals['macd'] = {
                    'signal': float(macd_signal),
                    'confidence': float(macd_conf),
                    'reason': f"MACD divergence {data['macd'][-1] - data['macd_signal'][-1]:.4f}"
                }
        
        # Volume
        if 'volume' in data and len(data['volume']) > 20:
            vol_mean = np.mean(data['volume'][-20:])
            if vol_mean > 0:
                vol_ratio = data['volume'][-1] / vol_mean
                vol_signal, vol_conf = self._calculate_volume_signal_fast(
                    vol_ratio, 1.2
                )
                signals['volume'] = {
                    'signal': float(vol_signal),
                    'confidence': float(vol_conf),
                    'reason': f"Volume {vol_ratio:.1f}x average"
                }
        
        return signals
    
    def batch_calculate_scores(self, symbols_data: Dict[str, Dict[str, Dict]]) -> Dict[str, TradingScore]:
        """
        Calcul en batch pour plusieurs symboles (optimisé pour le multi-threading)
        
        Args:
            symbols_data: Dict {symbol: signals_dict}
        
        Returns:
            Dict {symbol: TradingScore}
        """
        results = {}
        
        for symbol, signals in symbols_data.items():
            results[symbol] = self.calculate_score(signals, symbol)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Retourne les statistiques de performance"""
        cache_hits = sum(1 for _, score in self._cache.values() 
                        if score.metadata.get('cache_hit', False))
        total_calculations = len(self._cache)
        
        return {
            'cache_size': float(len(self._cache)),
            'cache_hit_rate': float(cache_hits) / float(total_calculations) if total_calculations > 0 else 0.0,
            'avg_calculation_time_ms': float(np.mean([
                score.metadata.get('calculation_time_ms', 0) 
                for _, score in self._cache.values()
            ])) if self._cache else 0.0
        }


# Fonction de migration pour remplacer facilement l'ancien engine
def migrate_to_optimized_engine(old_engine) -> OptimizedWeightedScoreEngine:
    """
    Migre depuis l'ancien WeightedScoreEngine vers la version optimisée
    
    Args:
        old_engine: Instance de WeightedScoreEngine
    
    Returns:
        OptimizedWeightedScoreEngine configuré identiquement
    """
    if hasattr(old_engine, 'weights'):
        return OptimizedWeightedScoreEngine(weights=old_engine.weights.copy())
    else:
        return OptimizedWeightedScoreEngine()


# Test de performance si exécuté directement
if __name__ == "__main__":
    import time
    
    print("Test de l'OptimizedWeightedScoreEngine...")
    
    # Créer l'engine
    engine = OptimizedWeightedScoreEngine()
    
    # Données de test
    test_signals = {
        'rsi': {'signal': 0.3, 'confidence': 0.8, 'reason': 'RSI oversold'},
        'bollinger': {'signal': 0.5, 'confidence': 0.7, 'reason': 'Near lower band'},
        'macd': {'signal': -0.2, 'confidence': 0.6, 'reason': 'Bearish crossover'},
        'volume': {'signal': 0.4, 'confidence': 0.9, 'reason': 'High volume'},
        'ma_cross': {'signal': 0.1, 'confidence': 0.5, 'reason': 'MA converging'},
        'momentum': {'signal': 0.0, 'confidence': 0.3, 'reason': 'Neutral'},
        'volatility': {'signal': -0.1, 'confidence': 0.4, 'reason': 'Low volatility'}
    }
    
    # Test de performance - Single calculation
    print("\n1. Test calcul unique (objectif < 0.5ms):")
    start = time.perf_counter()
    score = engine.calculate_score(test_signals, 'BTC/USDT')
    elapsed = (time.perf_counter() - start) * 1000
    print(f"   Calcul en {elapsed:.3f}ms")
    print(f"   Score: {score.total_score:.3f}, Confiance: {score.confidence:.2f}")
    print(f"   Direction: {score.direction}")
    
    # Test de performance - Batch
    print("\n2. Test batch 1000 calculs:")
    batch_data = {f'COIN{i}/USDT': test_signals for i in range(1000)}
    start = time.perf_counter()
    results = engine.batch_calculate_scores(batch_data)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"   1000 calculs en {elapsed:.1f}ms ({elapsed/1000:.3f}ms par calcul)")
    
    # Test avec arrays numpy
    print("\n3. Test avec données numpy (plus rapide):")
    numpy_data = {
        'close': np.random.uniform(45000, 55000, 100),
        'rsi': np.random.uniform(20, 80, 100),
        'bb_upper': np.random.uniform(51000, 52000, 100),
        'bb_lower': np.random.uniform(48000, 49000, 100),
        'macd': np.random.uniform(-0.5, 0.5, 100),
        'macd_signal': np.random.uniform(-0.3, 0.3, 100),
        'volume': np.random.uniform(50, 200, 100)
    }
    
    start = time.perf_counter()
    signals = engine.analyze_indicators_fast(numpy_data, 'BTC/USDT')
    score = engine.calculate_score(signals, 'BTC/USDT')
    elapsed = (time.perf_counter() - start) * 1000
    print(f"   Analyse + calcul en {elapsed:.3f}ms")
    
    # Stats de performance
    print("\n4. Statistiques de performance:")
    perf = engine.get_performance_stats()
    for key, value in perf.items():
        print(f"   {key}: {value:.3f}")