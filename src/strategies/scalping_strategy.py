# src/strategies/scalping_strategy.py
"""
Stratégie de scalping haute performance multi-signaux
Combine microstructure, momentum et mean-reversion
Latence cible: <20ms pour décision complète
"""

import numpy as np
from numba import jit, njit
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas_ta as ta
from collections import deque
import time
from ..core.fast_market_buffer import MarketSnapshot, FastMarketBuffer
from ..core.logger import log_info, log_debug
import pandas as pd


class SignalType(Enum):
    """Types de signaux de scalping"""
    LIQUIDITY_GRAB = "liquidity_grab"      # Absorption de liquidité
    MOMENTUM_BURST = "momentum_burst"        # Explosion de momentum
    MEAN_REVERSION = "mean_reversion"      # Retour à la moyenne
    SPREAD_ARBITRAGE = "spread_arb"        # Arbitrage de spread
    MICROSTRUCTURE = "microstructure"      # Pattern de microstructure
    BREAKOUT = "breakout"                  # Cassure de range


@dataclass
class ScalpingSignal:
    """Signal de scalping avec métadonnées"""
    signal_type: SignalType
    direction: str  # BUY/SELL
    confidence: float  # 0-1
    entry_price: float
    target_price: float
    stop_price: float
    size_multiplier: float  # Multiplicateur de taille de position
    ttl_ms: int  # Time to live en ms
    metadata: Dict


class ScalpingStrategy:
    """
    Stratégie de scalping multi-signaux haute performance
    Combine plusieurs techniques pour identifier des opportunités court terme
    """
    
    def __init__(self, symbol: str, config: Optional[Dict] = None):
        self.symbol = symbol
        self.config = config or self._default_config()
        
        # Buffers pour calculs
        self.price_buffer = deque(maxlen=500)
        self.volume_buffer = deque(maxlen=500)
        self.spread_buffer = deque(maxlen=200)
        self.trade_flow_buffer = deque(maxlen=100)
        
        # État interne
        self.last_signal_time = 0
        self.signal_cooldown_ms = 500  # Min 500ms entre signaux
        self.position_open = False
        
        # Paramètres adaptables
        self.volatility_regime = "NORMAL"
        self.liquidity_score = 1.0
        self.momentum_strength = 0.0

        self.fast_market_buffer = FastMarketBuffer()
        
        # Pré-calcul des seuils
        self._precompute_thresholds()
    
    def _default_config(self) -> Dict:
        """Configuration par défaut optimisée pour le scalping"""
        return {
            # Paramètres de base
            'min_spread_bps': 2,  # 2 basis points minimum
            'max_spread_bps': 20,  # 20 bps max
            'target_profit_bps': 10,  # 10 bps target
            'stop_loss_bps': 5,  # 5 bps stop
            
            # Microstructure
            'order_flow_imbalance_threshold': 0.7,
            'volume_spike_multiplier': 2.5,
            'liquidity_depth_levels': 5,
            
            # Mean reversion
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'rsi_period': 5,  # RSI court pour scalping
            'rsi_oversold': 20,
            'rsi_overbought': 80,
            
            # Momentum
            'momentum_lookback': 10,
            'momentum_threshold': 0.002,  # 0.2%
            'volume_momentum_correlation': 0.6,
            
            # Risk
            'max_position_time_seconds': 300,  # 5 minutes max
            'dynamic_sizing': True,
            'volatility_adjustment': True
        }
    
    def _precompute_thresholds(self):
        """Pré-calcule les seuils pour optimiser les performances"""
        self.spread_threshold = self.config['min_spread_bps'] / 10000
        self.profit_target = self.config['target_profit_bps'] / 10000
        self.stop_loss = self.config['stop_loss_bps'] / 10000
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _calculate_order_flow_imbalance(bid_sizes: np.ndarray, 
                                       ask_sizes: np.ndarray) -> float:
        """
        Calcul optimisé du déséquilibre du carnet d'ordres
        Retourne une valeur entre -1 (pression vendeuse) et 1 (pression acheteuse)
        """
        total_bid = np.sum(bid_sizes)
        total_ask = np.sum(ask_sizes)
        
        if total_bid + total_ask == 0:
            return 0.0
        
        imbalance = (total_bid - total_ask) / (total_bid + total_ask)
        return np.clip(imbalance, -1.0, 1.0)
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _detect_liquidity_grab(prices: np.ndarray, volumes: np.ndarray,
                              window: int = 20) -> Tuple[bool, float]:
        """
        Détecte une absorption de liquidité (gros ordre qui "mange" le carnet)
        Retourne (signal_detected, confidence)
        """
        if len(prices) < window:
            return False, 0.0
        
        # Volume spike
        recent_vol = volumes[-window:]
        avg_vol = np.mean(recent_vol[:-1])
        last_vol = recent_vol[-1]
        
        if avg_vol == 0:
            return False, 0.0
        
        vol_ratio = last_vol / avg_vol
        
        # Price movement minimal malgré gros volume = absorption
        price_change = abs(prices[-1] - prices[-2]) / prices[-2]
        
        if vol_ratio > 3.0 and price_change < 0.0005:  # <0.05% move
            confidence = min(vol_ratio / 5.0, 1.0)  # Max confidence at 5x volume
            return True, confidence
        
        return False, 0.0
    
    def _analyze_microstructure(self, snapshot: MarketSnapshot, 
                               stats: Dict) -> Optional[ScalpingSignal]:
        """Analyse la microstructure du marché pour signaux courts terme"""
        
        # Vérifier le spread
        spread_bps = (snapshot.spread / snapshot.mid) * 10000
        if spread_bps > self.config['max_spread_bps']:
            return None
        
        # Analyser le déséquilibre
        if snapshot.bid_size > 0 and snapshot.ask_size > 0:
            imbalance = self._calculate_order_flow_imbalance(
                np.array([snapshot.bid_size]),
                np.array([snapshot.ask_size])
            )
            
            # Signal fort de déséquilibre
            if abs(imbalance) > self.config['order_flow_imbalance_threshold']:
                direction = "BUY" if imbalance > 0 else "SELL"
                confidence = abs(imbalance)
                
                # Calcul des prix
                if direction == "BUY":
                    entry = snapshot.ask  # Acheter au ask
                    target = entry * (1 + self.profit_target)
                    stop = entry * (1 - self.stop_loss)
                else:
                    entry = snapshot.bid  # Vendre au bid
                    target = entry * (1 - self.profit_target)
                    stop = entry * (1 + self.stop_loss)
                
                return ScalpingSignal(
                    signal_type=SignalType.MICROSTRUCTURE,
                    direction=direction,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target,
                    stop_price=stop,
                    size_multiplier=1.0 + confidence,  # Augmenter taille si confiant
                    ttl_ms=2000,  # 2 secondes
                    metadata={
                        'imbalance': imbalance,
                        'spread_bps': spread_bps
                    }
                )
        
        return None
    
    def _analyze_momentum(self, buffer: FastMarketBuffer) -> Optional[ScalpingSignal]:
        """Détecte les explosions de momentum pour surfer la vague"""
        
        if len(self.price_buffer) < self.config['momentum_lookback']:
            return None
        
        # Convertir en arrays numpy
        prices = np.array(list(self.price_buffer))
        volumes = np.array(list(self.volume_buffer))
        
        # Momentum sur N périodes
        momentum = (prices[-1] - prices[-self.config['momentum_lookback']]) / prices[-self.config['momentum_lookback']]
        
        # Volume confirmant le momentum
        vol_increase = volumes[-1] / np.mean(volumes[:-1])
        
        if abs(momentum) > self.config['momentum_threshold'] and vol_increase > 1.5:
            # Corrélation volume/prix
            if len(prices) > 20:
                price_returns = np.diff(prices[-20:]) / prices[-20:-1]
                vol_changes = np.diff(volumes[-20:]) / volumes[-20:-1]
                correlation = np.corrcoef(price_returns, vol_changes)[0, 1]
            else:
                correlation = 0.5
            
            if abs(correlation) > self.config['volume_momentum_correlation']:
                direction = "BUY" if momentum > 0 else "SELL"
                confidence = min(abs(momentum) * 100, 1.0) * min(vol_increase / 2, 1.0)
                
                # Entrée immédiate au marché
                current_price = prices[-1]
                if direction == "BUY":
                    entry = current_price * 1.0001  # Petit slippage
                    target = entry * (1 + self.profit_target * 1.5)  # Target plus large
                    stop = entry * (1 - self.stop_loss * 0.7)  # Stop plus serré
                else:
                    entry = current_price * 0.9999
                    target = entry * (1 - self.profit_target * 1.5)
                    stop = entry * (1 + self.stop_loss * 0.7)
                
                return ScalpingSignal(
                    signal_type=SignalType.MOMENTUM_BURST,
                    direction=direction,
                    confidence=confidence,
                    entry_price=entry,
                    target_price=target,
                    stop_price=stop,
                    size_multiplier=0.8,  # Taille réduite car plus risqué
                    ttl_ms=1000,  # 1 seconde seulement
                    metadata={
                        'momentum': momentum,
                        'volume_spike': vol_increase,
                        'correlation': correlation
                    }
                )
        
        return None
    
    def _analyze_mean_reversion(self) -> Optional[ScalpingSignal]:
        """Identifie les opportunités de retour à la moyenne"""
        
        if len(self.price_buffer) < self.config['bollinger_period']:
            return None
        df = pd.DataFrame({'close': list(self.price_buffer)})
        bbands = df.ta.bbands(length=self.config['bollinger_period'], std=self.config['bollinger_std'])
        df['rsi'] = df.ta.rsi(length=self.config['rsi_period'])
        if any(col not in bbands or bbands[col].notnull().sum() == 0 for col in [f'BBL_{self.config["bollinger_period"]}_2.0', f'BBU_{self.config["bollinger_period"]}_2.0']):
            return None
        lower_band = bbands[f'BBL_{self.config["bollinger_period"]}_2.0'].iloc[-1]
        upper_band = bbands[f'BBU_{self.config["bollinger_period"]}_2.0'].iloc[-1]
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1] if 'rsi' in df and df['rsi'].notnull().sum() > 0 else 0
        if current_price < lower_band and rsi < self.config['rsi_oversold']:
            confidence = (self.config['rsi_oversold'] - rsi) / self.config['rsi_oversold']
            confidence *= (lower_band - current_price) / lower_band
            entry = current_price * 1.0002
            target = (upper_band + lower_band) / 2  # moyenne
            stop = current_price * 0.995
            return ScalpingSignal(
                signal_type=SignalType.MEAN_REVERSION,
                direction="BUY",
                confidence=min(confidence * 2, 1.0),
                entry_price=entry,
                target_price=target,
                stop_price=stop,
                size_multiplier=1.2,
                ttl_ms=5000,
                metadata={'rsi': rsi, 'bollinger_position': 'below_lower'}
            )
        elif current_price > upper_band and rsi > self.config['rsi_overbought']:
            confidence = (rsi - self.config['rsi_overbought']) / (100 - self.config['rsi_overbought'])
            confidence *= (current_price - upper_band) / upper_band
            entry = current_price * 0.9998
            target = (upper_band + lower_band) / 2
            stop = current_price * 1.005
            return ScalpingSignal(
                signal_type=SignalType.MEAN_REVERSION,
                direction="SELL",
                confidence=min(confidence * 2, 1.0),
                entry_price=entry,
                target_price=target,
                stop_price=stop,
                size_multiplier=1.2,
                ttl_ms=5000,
                metadata={'rsi': rsi, 'bollinger_position': 'above_upper'}
            )
        return None
    
    def generate_signal(self, snapshot: MarketSnapshot, stats: Dict,
                       microstructure: Dict) -> Optional[ScalpingSignal]:
        """
        Génère un signal de trading en combinant toutes les analyses
        Latence cible: <20ms
        """
        start_time = time.time()
        
        # Vérifier le cooldown
        current_time = int(time.time() * 1000)
        if current_time - self.last_signal_time < self.signal_cooldown_ms:
            return None
        
        # Mettre à jour les buffers
        self.price_buffer.append(snapshot.last)
        self.volume_buffer.append(snapshot.volume)
        self.spread_buffer.append(snapshot.spread)
        
        # Analyser en parallèle (dans l'ordre de rapidité)
        signals = []
        
        # 1. Microstructure (le plus rapide)
        micro_signal = self._analyze_microstructure(snapshot, stats)
        if micro_signal:
            signals.append(micro_signal)
        
        # 2. Momentum
        momentum_signal = self._analyze_momentum(self.fast_market_buffer)
        if momentum_signal:
            signals.append(momentum_signal)
        
        # 3. Mean reversion (le plus lent à cause de talib)
        if len(signals) == 0:  # Seulement si pas déjà de signal
            mr_signal = self._analyze_mean_reversion()
            if mr_signal:
                signals.append(mr_signal)
        
        # Sélectionner le meilleur signal
        if signals:
            # Trier par confiance
            best_signal = max(signals, key=lambda s: s.confidence)
            
            # Ajuster selon le régime de marché
            best_signal = self._adjust_for_market_regime(best_signal)
            
            self.last_signal_time = current_time
            
            # Mesurer la latence
            latency = (time.time() - start_time) * 1000
            log_debug(f"Signal généré en {latency:.2f}ms - Type: {best_signal.signal_type.value}")
            
            return best_signal
        
        return None
    
    def _adjust_for_market_regime(self, signal: ScalpingSignal) -> ScalpingSignal:
        """Ajuste le signal selon le régime de marché actuel"""
        
        # En régime volatile, réduire la taille et élargir les stops
        if self.volatility_regime == "HIGH":
            signal.size_multiplier *= 0.5
            signal.stop_price = signal.entry_price * (1 - self.stop_loss * 1.5) if signal.direction == "BUY" \
                               else signal.entry_price * (1 + self.stop_loss * 1.5)
        
        # En régime calme, on peut être plus agressif
        elif self.volatility_regime == "LOW":
            signal.size_multiplier *= 1.5
            signal.ttl_ms = int(signal.ttl_ms * 1.5)  # Plus de temps
        
        return signal
    
    def update_market_regime(self, volatility: float, liquidity: float):
        """Met à jour le régime de marché pour adapter la stratégie"""
        
        # Régime de volatilité
        if volatility > 0.02:  # 2% 
            self.volatility_regime = "HIGH"
        elif volatility < 0.005:  # 0.5%
            self.volatility_regime = "LOW"
        else:
            self.volatility_regime = "NORMAL"
        
        # Score de liquidité
        self.liquidity_score = liquidity
        
        # Ajuster les paramètres
        if self.volatility_regime == "HIGH":
            self.signal_cooldown_ms = 1000  # Plus de cooldown
            self.config['min_spread_bps'] = 5  # Accepter des spreads plus larges
        else:
            self.signal_cooldown_ms = 300
            self.config['min_spread_bps'] = 2