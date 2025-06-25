# src/strategies/ai_enhanced_strategy.py
"""
Stratégie hybride IA + Analyse technique avancée
Combine ML, sentiment analysis et patterns complexes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Changer pour Classifier
from sklearn.preprocessing import StandardScaler
import warnings

import pandas_ta as ta

from datetime import datetime, timedelta
import asyncio

from ..core.logger import log_info, log_debug, log_error
from ..core.weighted_score_engine import WeightedScoreEngine, TradingScore
from .strategy import Strategy


@dataclass
class MarketRegime:
    """Régime de marché identifié"""
    type: str  # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
    strength: float  # 0-1
    confidence: float  # 0-1
    volatility: float
    volume_profile: str  # INCREASING, DECREASING, STABLE
    

class AIEnhancedStrategy(Strategy):
    """
    Stratégie avancée combinant :
    - Machine Learning pour prédiction
    - Analyse de régimes de marché
    - Patterns complexes (Harmonic, Elliott Waves simplifiés)
    - Analyse multi-timeframe
    - Sentiment de marché via volume/momentum
    """
    
    def __init__(self, symbol: str, config: Optional[Dict] = None):
        super().__init__(symbol)
        self.config = config or self._default_config()
        
        # Modèles ML
        self.price_predictor = None  # Prédit le mouvement de prix
        self.regime_classifier = None  # Identifie le régime de marché
        self.risk_assessor = None  # Évalue le risque du trade
        
        # Scalers pour normalisation
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        # État du marché
        self.current_regime = None
        self.regime_history = []
        
        # Buffers pour features
        self.feature_buffer = []
        self.prediction_buffer = []
        
        # Score engine avec poids dynamiques
        self.score_engine = WeightedScoreEngine()
        
        log_info(f"AIEnhancedStrategy initialisée pour {symbol}")
    
    def _default_config(self) -> Dict:
        return {
            # ML
            'use_ml_prediction': True,
            'ml_confidence_threshold': 0.65,
            'retrain_interval_hours': 24,
            
            # Analyse technique avancée
            'use_harmonic_patterns': True,
            'use_volume_profile': True,
            'use_order_flow': True,
            
            # Multi-timeframe
            'timeframes': ['5m', '15m', '1h', '4h'],
            'mtf_alignment_required': False,
            
            # Régimes de marché
            'adapt_to_regime': True,
            'regime_lookback_periods': 100,
            
            # Risk
            'base_risk_per_trade': 0.01,  # 1%
            'max_risk_per_trade': 0.02,   # 2%
            'risk_multiplier_by_confidence': True,
            
            # Patterns
            'min_pattern_confidence': 0.7,
            'pattern_types': ['double_top', 'double_bottom', 'triangle', 
                            'flag', 'wedge', 'head_shoulders']
        }
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extrait des features avancées pour le ML
        ~50-100 features incluant indicateurs, patterns, microstructure
        """
        features = []
        
        # Prix et rendements
        close = pd.Series(df['close']) if not isinstance(df['close'], pd.Series) else df['close']
        high = pd.Series(df['high']) if not isinstance(df['high'], pd.Series) else df['high']
        low = pd.Series(df['low']) if not isinstance(df['low'], pd.Series) else df['low']
        volume = pd.Series(df['volume']) if not isinstance(df['volume'], pd.Series) else df['volume']
        features = []
        current_price = close.iloc[-1] if len(close) > 0 else 0
        features.append(current_price)
        features.extend([
            close.pct_change().iloc[-1] if len(close) > 1 else 0,
            close.pct_change(5).iloc[-1] if len(close) > 5 else 0,
            close.pct_change(20).iloc[-1] if len(close) > 20 else 0,
            (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1] if len(high) > 0 and len(low) > 0 and close.iloc[-1] != 0 else 0,
        ])
        
        # Indicateurs techniques classiques
        # RSI
        df['rsi'] = df.ta.rsi(length=14)
        features.append(df['rsi'].iloc[-1] if 'rsi' in df and df['rsi'].notnull().sum() > 0 else 0)
        
        # MACD
        macd = df.ta.macd()
        for col in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']:
            features.append(macd[col].iloc[-1] if col in macd and macd[col] is not None and macd[col].notnull().sum() > 0 else 0)
        
        # Bollinger Bands
        bbands = df.ta.bbands(length=20, std=2)
        if all(col in bbands and bbands[col] is not None and bbands[col].notnull().sum() > 0 for col in ['BBL_20_2.0', 'BBU_20_2.0']):
            lower = bbands['BBL_20_2.0'].iloc[-1]
            upper = bbands['BBU_20_2.0'].iloc[-1]
            features.append((current_price - lower) / (upper - lower) if (upper - lower) != 0 else 0)
        else:
            features.append(0)
        
        # ADX
        adx = df.ta.adx()
        features.append(adx['ADX_14'].iloc[-1] if 'ADX_14' in adx and adx['ADX_14'] is not None and adx['ADX_14'].notnull().sum() > 0 else 0)
        
        # Volume analysis
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
        if isinstance(volume, pd.Series) and len(volume) > 0:
            volume_sma = volume.rolling(20).mean()
            if isinstance(volume_sma, np.ndarray):
                volume_sma = pd.Series(volume_sma)
            volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1] if len(volume) > 0 and len(volume_sma) > 0 and volume_sma.iloc[-1] != 0 else 0
            features.append(volume_ratio)
        else:
            features.append(0)
        
        # OBV
        obv = ta.obv(close, volume)
        if obv is not None and isinstance(obv, np.ndarray):
            obv = pd.Series(obv)
        if isinstance(obv, pd.Series) and obv.notnull().sum() > 0:
            obv_val = obv.iloc[-1]
            obv_20 = obv.iloc[-20] if len(obv) > 20 else obv.iloc[0]
            obv_change = (obv_val - obv_20) / obv_20 if obv_20 != 0 else 0
            features.append(obv_change)
        else:
            features.append(0)
        
        # Microstructure
        if 'bid' in df.columns and 'ask' in df.columns:
            bid = df['bid']
            ask = df['ask']
            if bid is not None and isinstance(bid, np.ndarray):
                bid = pd.Series(bid)
            if ask is not None and isinstance(ask, np.ndarray):
                ask = pd.Series(ask)
            spread_pct = ((ask - bid) / close * 100).mean() if bid is not None and ask is not None and len(ask) > 0 and len(bid) > 0 else 0
            features.append(spread_pct)
        
        # Volatilité (ATR)
        atr = df.ta.atr()
        if atr is not None and isinstance(atr, np.ndarray):
            atr = pd.Series(atr)
        features.append(atr.iloc[-1] if isinstance(atr, pd.Series) and atr.notnull().sum() > 0 else 0)
        
        # Patterns détectés (simplifié)
        features.extend(self._detect_patterns_features(df))
        
        # Structure du marché
        # Support/Resistance levels
        support, resistance = self._calculate_support_resistance(df)
        distance_to_support = (current_price - support) / current_price if current_price != 0 else 0
        distance_to_resistance = (resistance - current_price) / current_price if current_price != 0 else 0
        features.extend([distance_to_support, distance_to_resistance])
        
        # Momentum
        roc_5 = ta.roc(close, length=5)
        roc_10 = ta.roc(close, length=10)
        if roc_5 is not None and isinstance(roc_5, np.ndarray):
            roc_5 = pd.Series(roc_5)
        if roc_10 is not None and isinstance(roc_10, np.ndarray):
            roc_10 = pd.Series(roc_10)
        features.append(roc_5.iloc[-1] if isinstance(roc_5, pd.Series) and roc_5.notnull().sum() > 0 else 0)
        features.append(roc_10.iloc[-1] if isinstance(roc_10, pd.Series) and roc_10.notnull().sum() > 0 else 0)
        
        # Market regime features
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        if sma_20 is not None and isinstance(sma_20, np.ndarray):
            sma_20 = pd.Series(sma_20)
        if sma_50 is not None and isinstance(sma_50, np.ndarray):
            sma_50 = pd.Series(sma_50)
        trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] if isinstance(sma_20, pd.Series) and isinstance(sma_50, pd.Series) and sma_50.iloc[-1] != 0 else 0
        features.append(trend_strength)
        
        return np.array(features)
    
    def _detect_patterns_features(self, df: pd.DataFrame) -> List[float]:
        """Détecte des patterns et retourne des features binaires/probabilités"""
        features = []
        
        # Double Top/Bottom (simplifié)
        highs = df['high'].rolling(20).max()
        lows = df['low'].rolling(20).min()
        
        # Double top
        recent_highs = df['high'].iloc[-20:]
        peaks = (recent_highs == recent_highs.max()).sum()
        double_top_score = 1.0 if peaks >= 2 else 0.0
        features.append(double_top_score)
        
        # Breakout potential
        current_price = df['close'].iloc[-1]
        range_high = df['high'].iloc[-50:].max()
        range_low = df['low'].iloc[-50:].min()
        breakout_score = 0.0
        if current_price > range_high * 0.98:
            breakout_score = 1.0
        elif current_price < range_low * 1.02:
            breakout_score = -1.0
        features.append(breakout_score)
        
        return features
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calcule les niveaux de support et résistance dynamiques"""
        # Méthode simple : utiliser les fractals
        window = 10
        
        # Resistance : plus haut des highs récents qui ont été touchés plusieurs fois
        highs = df['high'].iloc[-50:]
        resistance = highs.nlargest(5).mean()
        
        # Support : plus bas des lows récents qui ont été touchés plusieurs fois
        lows = df['low'].iloc[-50:]
        support = lows.nsmallest(5).mean()
        
        return support, resistance
    
    def _identify_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Identifie le régime de marché actuel avec ML ou règles"""
        adx = ta.adx(df['high'], df['low'], df['close'], timeperiod=14)
        close = pd.Series(df['close']) if not isinstance(df['close'], pd.Series) else df['close']
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        returns = close.pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        volume = pd.Series(df['volume']) if not isinstance(df['volume'], pd.Series) else df['volume']
        volume_sma = volume.rolling(20).mean()
        volume_trend = "STABLE"
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
        if isinstance(volume_sma, np.ndarray):
            volume_sma = pd.Series(volume_sma)
        if len(volume) > 5 and len(volume_sma) > 0 and volume_sma.iloc[-1] != 0:
            if volume.iloc[-5:].mean() > volume_sma.iloc[-1] * 1.2:
                volume_trend = "INCREASING"
            elif volume.iloc[-5:].mean() < volume_sma.iloc[-1] * 0.8:
                volume_trend = "DECREASING"
        regime_type = "RANGING"
        strength = 0.5
        adx_val = 0
        if adx is not None and hasattr(adx, 'columns') and 'ADX_14' in adx and adx['ADX_14'] is not None and isinstance(adx['ADX_14'], (pd.Series, np.ndarray)):
            adx_col = adx['ADX_14']
            if isinstance(adx_col, np.ndarray):
                adx_col = pd.Series(adx_col)
            if adx_col.notnull().sum() > 0:
                adx_val = adx_col.iloc[-1]
        if adx_val > 25:
            if sma_20 is not None and sma_50 is not None and isinstance(sma_20, pd.Series) and isinstance(sma_50, pd.Series) and sma_20.iloc[-1] > sma_50.iloc[-1]:
                regime_type = "TRENDING_UP"
                strength = min(adx_val / 50, 1.0)
            else:
                regime_type = "TRENDING_DOWN"
                strength = min(adx_val / 50, 1.0)
        elif volatility is not None and isinstance(volatility, pd.Series) and len(volatility) > 0 and volatility.iloc[-1] > 0.4:
            regime_type = "VOLATILE"
            strength = min(volatility.iloc[-1], 1.0)
        return MarketRegime(
            type=regime_type,
            strength=strength,
            confidence=0.8,
            volatility=volatility.iloc[-1] if volatility is not None and isinstance(volatility, pd.Series) and len(volatility) > 0 else 0,
            volume_profile=volume_trend
        )
    
    async def train_models(self, historical_data: pd.DataFrame):
        """Entraîne les modèles ML sur données historiques"""
        log_info("Début de l'entraînement des modèles ML...")
        
        # Préparer les features et labels
        X, y_price, y_risk = [], [], []
        
        for i in range(100, len(historical_data) - 10):
            # Features
            window = historical_data.iloc[i-100:i]
            features = self._extract_features(window)
            X.append(features)
            
            # Label prix : direction dans les 10 prochaines périodes
            future_return = (historical_data['close'].iloc[i+10] - historical_data['close'].iloc[i]) / historical_data['close'].iloc[i]
            y_price.append(1 if future_return > 0.005 else 0)  # 0.5% threshold
            
            # Label risque : drawdown max dans les 10 prochaines périodes
            future_prices = historical_data['close'].iloc[i:i+10]
            max_drawdown = (future_prices.min() - historical_data['close'].iloc[i]) / historical_data['close'].iloc[i]
            y_risk.append(1 if max_drawdown < -0.02 else 0)  # 2% drawdown = risqué
        
        X = np.array(X)
        y_price = np.array(y_price)
        y_risk = np.array(y_risk)
        
        # Normaliser les features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Entraîner le prédicteur de prix
        self.price_predictor = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.price_predictor.fit(X_scaled, y_price)
        
        # Entraîner l'évaluateur de risque
        self.risk_assessor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.risk_assessor.fit(X_scaled, y_risk)
        
        self.is_trained = True
        log_info("✅ Modèles ML entraînés avec succès")
    
    def should_enter(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Décision d'entrée basée sur ML + analyse technique avancée
        """
        if len(df) < 100:
            return None
        
        # 1. Identifier le régime de marché
        if self.current_regime is None:
            self.current_regime = self._identify_market_regime(df)
        regime_type = self.current_regime.type
        log_debug(f"Régime: {regime_type} (force: {self.current_regime.strength:.2f})")
        
        # 2. Extraire les features
        features = self._extract_features(df)
        
        # 3. Prédictions ML (si entraîné)
        ml_signal = 0
        ml_confidence = 0.5
        
        if self.is_trained and self.price_predictor is not None:
            features_scaled = self.feature_scaler.transform([features]) if self.feature_scaler else [features]
            
            # Prédiction de direction
            price_pred = self.price_predictor.predict_proba(features_scaled)[0]
            ml_signal = price_pred[1] - 0.5  # Centré sur 0
            
            # Évaluation du risque
            risk_score = 0.5
            if self.risk_assessor is not None:
                risk_pred = self.risk_assessor.predict_proba(features_scaled)[0]
                risk_score = risk_pred[1]
            
            # Ajuster la confiance selon le risque
            ml_confidence = price_pred[1] * (1 - risk_score * 0.5)
        
        # 4. Analyse technique avancée
        # Score des indicateurs classiques
        tech_signals = self._analyze_technical_indicators(df)
        
        # 5. Analyse des patterns
        pattern_signal = self._analyze_patterns(df)
        
        # 6. Score final pondéré selon le régime
        weights = self._get_regime_weights(self.current_regime)
        
        final_score = (
            weights['ml'] * ml_signal +
            weights['technical'] * tech_signals['score'] +
            weights['pattern'] * pattern_signal['score']
        )
        
        final_confidence = (
            weights['ml'] * ml_confidence +
            weights['technical'] * tech_signals['confidence'] +
            weights['pattern'] * pattern_signal['confidence']
        )
        
        # 7. Décision finale
        if final_score > 0.3 and final_confidence > 0.6:
            current_price = df['close'].iloc[-1] if hasattr(df['close'], 'iloc') and len(df['close']) > 0 else 0
            atr = ta.atr(df['high'], df['low'], df['close'])
            if atr is not None and isinstance(atr, np.ndarray):
                atr = pd.Series(atr)
            atr_val = atr.iloc[-1] if isinstance(atr, pd.Series) and atr.notnull().sum() > 0 else 0
            stop_distance = atr_val * 2
            target_distance = atr_val * 3
            if regime_type == "VOLATILE":
                stop_distance *= 1.5
                target_distance *= 1.5
            return {
                'action': 'BUY',
                'confidence': final_confidence,
                'score': final_score,
                'entry_price': current_price,
                'stop_loss': current_price - stop_distance,
                'take_profit': current_price + target_distance,
                'size_multiplier': self._calculate_position_size_multiplier(final_confidence),
                'regime': regime_type,
                'reason': self._generate_entry_reason(tech_signals, pattern_signal, ml_signal)
            }
        
        return None
    
    def _analyze_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Analyse technique multi-indicateurs avec scoring"""
        signals = []
        
        # RSI
        rsi = ta.rsi(df['close'])
        if rsi is not None and isinstance(rsi, np.ndarray):
            rsi = pd.Series(rsi)
        rsi_val = rsi.iloc[-1] if isinstance(rsi, pd.Series) and rsi.notnull().sum() > 0 else 0
        if rsi_val < 30:
            signals.append({'name': 'RSI_oversold', 'score': 1.0, 'confidence': 0.9})
        elif rsi_val > 70:
            signals.append({'name': 'RSI_overbought', 'score': -1.0, 'confidence': 0.9})
        
        # MACD
        macd = ta.macd(df['close'])
        macd_hist = None
        if macd is not None and hasattr(macd, 'columns') and 'MACDh_12_26_9' in macd and macd['MACDh_12_26_9'] is not None:
            macd_hist = macd['MACDh_12_26_9']
            if isinstance(macd_hist, np.ndarray):
                macd_hist = pd.Series(macd_hist)
        if macd_hist is not None and macd_hist.notnull().sum() > 1:
            if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
                signals.append({'name': 'MACD_bullish_cross', 'score': 1.0, 'confidence': 0.8})
        
        # Bollinger Bands
        bb = ta.bbands(df['close'])
        bb_lower = None
        if bb is not None and hasattr(bb, 'columns') and 'BBL_20_2.0' in bb and bb['BBL_20_2.0'] is not None:
            bb_lower = bb['BBL_20_2.0']
            if isinstance(bb_lower, np.ndarray):
                bb_lower = pd.Series(bb_lower)
        if bb_lower is not None and bb_lower.notnull().sum() > 0:
            if df['close'].iloc[-1] < bb_lower.iloc[-1]:
                signals.append({'name': 'BB_oversold', 'score': 1.0, 'confidence': 0.7})
        
        # Agrégation
        if signals:
            avg_score = np.mean([s['score'] for s in signals])
            avg_confidence = np.mean([s['confidence'] for s in signals])
            return {'score': avg_score, 'confidence': avg_confidence, 'signals': signals}
        
        return {'score': 0, 'confidence': 0, 'signals': []}
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """Détection de patterns chartistes"""
        # Simplifié pour l'exemple
        pattern_score = 0
        pattern_confidence = 0
        patterns_found = []
        
        # Double bottom
        lows = df['low'].iloc[-30:]
        if len(lows) >= 30:
            first_bottom = lows.iloc[:15].min()
            second_bottom = lows.iloc[15:].min()
            
            if abs(first_bottom - second_bottom) / first_bottom < 0.02:  # 2% de différence
                pattern_score = 1.0
                pattern_confidence = 0.8
                patterns_found.append('double_bottom')
        
        return {
            'score': pattern_score,
            'confidence': pattern_confidence,
            'patterns': patterns_found
        }
    
    def _get_regime_weights(self, regime: MarketRegime) -> Dict:
        """Poids adaptatifs selon le régime de marché"""
        if regime.type == "TRENDING_UP":
            return {'ml': 0.3, 'technical': 0.5, 'pattern': 0.2}
        elif regime.type == "TRENDING_DOWN":
            return {'ml': 0.4, 'technical': 0.4, 'pattern': 0.2}
        elif regime.type == "VOLATILE":
            return {'ml': 0.5, 'technical': 0.3, 'pattern': 0.2}
        else:  # RANGING
            return {'ml': 0.2, 'technical': 0.3, 'pattern': 0.5}
    
    def _calculate_position_size_multiplier(self, confidence: float) -> float:
        """Calcule le multiplicateur de taille selon la confiance"""
        if confidence > 0.85:
            return 1.5
        elif confidence > 0.75:
            return 1.2
        elif confidence > 0.65:
            return 1.0
        else:
            return 0.7
    
    def _generate_entry_reason(self, tech_signals: Dict, pattern_signal: Dict, 
                              ml_signal: float) -> str:
        """Génère une explication détaillée du signal"""
        reasons = []
        
        if ml_signal > 0.3:
            reasons.append(f"ML bullish ({ml_signal:.2f})")
        
        if tech_signals['signals']:
            tech_names = [s['name'] for s in tech_signals['signals']]
            reasons.append(f"Tech: {', '.join(tech_names)}")
        
        if pattern_signal['patterns']:
            reasons.append(f"Patterns: {', '.join(pattern_signal['patterns'])}")
        
        if self.current_regime is not None:
            reasons.append(f"Régime: {self.current_regime.type}")
        
        return " | ".join(reasons)
    
    def should_exit(self, df: pd.DataFrame, position: Dict) -> Optional[Dict]:
        """
        Détermine si on doit sortir de position (logique complète)
        Args:
            df: DataFrame avec les données de marché
            position: Dictionnaire avec les infos de la position actuelle
        Returns:
            Dictionnaire avec le signal de sortie ou None
        """
        if len(df) < 20 or not position:
            return None

        current_price = df['close'].iloc[-1]
        side = position.get('side', '').upper()
        stop_loss = position.get('stop_loss')
        take_profit = position.get('take_profit')

        # Vérifier stop loss et take profit
        if side == 'LONG':
            if stop_loss and current_price <= stop_loss:
                return {
                    'action': 'SELL',
                    'type': 'market',
                    'reason': 'Stop loss atteint',
                }
            if take_profit and current_price >= take_profit:
                return {
                    'action': 'SELL',
                    'type': 'market',
                    'reason': 'Take profit atteint',
                }
        elif side == 'SHORT':
            if stop_loss and current_price >= stop_loss:
                return {
                    'action': 'BUY',
                    'type': 'market',
                    'reason': 'Stop loss atteint',
                }
            if take_profit and current_price <= take_profit:
                return {
                    'action': 'BUY',
                    'type': 'market',
                    'reason': 'Take profit atteint',
                }

        # Analyse technique/patterns/ML pour sortie anticipée
        tech_signals = self._analyze_technical_indicators(df)
        pattern_signal = self._analyze_patterns(df)
        weights = self._get_regime_weights(self.current_regime or self._identify_market_regime(df))
        final_score = (
            weights['ml'] * 0 +
            weights['technical'] * tech_signals['score'] +
            weights['pattern'] * pattern_signal['score']
        )
        exit_threshold = -0.3
        if side == 'LONG' and final_score < exit_threshold:
            return {
                'action': 'SELL',
                'type': 'market',
                'reason': f'Score technique bas: {final_score:.2f}',
            }
        if side == 'SHORT' and final_score > -exit_threshold:
            return {
                'action': 'BUY',
                'type': 'market',
                'reason': f'Score technique élevé: {final_score:.2f}',
            }
        return None