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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import talib
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
    
    def __init__(self, symbol: str, config: Dict = None):
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
        features.extend([
            df['close'].iloc[-1],
            df['close'].pct_change().iloc[-1],
            df['close'].pct_change(5).iloc[-1],
            df['close'].pct_change(20).iloc[-1],
            (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1],  # Range %
        ])
        
        # Indicateurs techniques classiques
        # RSI multi-périodes
        for period in [7, 14, 21]:
            rsi = talib.RSI(df['close'], timeperiod=period)
            features.append(rsi.iloc[-1])
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'])
        features.extend([macd.iloc[-1], signal.iloc[-1], hist.iloc[-1]])
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'])
        bb_position = (df['close'].iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        features.append(bb_position)
        
        # ADX (force de tendance)
        adx = talib.ADX(df['high'], df['low'], df['close'])
        features.append(adx.iloc[-1])
        
        # Volume analysis
        volume_sma = df['volume'].rolling(20).mean()
        volume_ratio = df['volume'].iloc[-1] / volume_sma.iloc[-1]
        features.append(volume_ratio)
        
        # OBV (On Balance Volume)
        obv = talib.OBV(df['close'], df['volume'])
        obv_change = (obv.iloc[-1] - obv.iloc[-20]) / obv.iloc[-20] if obv.iloc[-20] != 0 else 0
        features.append(obv_change)
        
        # Microstructure
        # Spread moyen (si disponible)
        if 'bid' in df.columns and 'ask' in df.columns:
            spread_pct = ((df['ask'] - df['bid']) / df['close'] * 100).mean()
            features.append(spread_pct)
        
        # Volatilité (ATR)
        atr = talib.ATR(df['high'], df['low'], df['close'])
        atr_pct = (atr.iloc[-1] / df['close'].iloc[-1]) * 100
        features.append(atr_pct)
        
        # Patterns détectés (simplifié)
        features.extend(self._detect_patterns_features(df))
        
        # Structure du marché
        # Support/Resistance levels
        support, resistance = self._calculate_support_resistance(df)
        distance_to_support = (df['close'].iloc[-1] - support) / df['close'].iloc[-1]
        distance_to_resistance = (resistance - df['close'].iloc[-1]) / df['close'].iloc[-1]
        features.extend([distance_to_support, distance_to_resistance])
        
        # Momentum
        roc_5 = talib.ROC(df['close'], timeperiod=5)
        roc_10 = talib.ROC(df['close'], timeperiod=10)
        features.extend([roc_5.iloc[-1], roc_10.iloc[-1]])
        
        # Market regime features
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
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
        # ADX pour la force de tendance
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Direction de tendance
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        # Volatilité
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualisée
        
        # Volume trend
        volume_sma = df['volume'].rolling(20).mean()
        volume_trend = "STABLE"
        if df['volume'].iloc[-5:].mean() > volume_sma.iloc[-1] * 1.2:
            volume_trend = "INCREASING"
        elif df['volume'].iloc[-5:].mean() < volume_sma.iloc[-1] * 0.8:
            volume_trend = "DECREASING"
        
        # Classification du régime
        regime_type = "RANGING"
        strength = 0.5
        
        if adx.iloc[-1] > 25:  # Tendance forte
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                regime_type = "TRENDING_UP"
                strength = min(adx.iloc[-1] / 50, 1.0)
            else:
                regime_type = "TRENDING_DOWN"
                strength = min(adx.iloc[-1] / 50, 1.0)
        elif volatility.iloc[-1] > 0.4:  # Haute volatilité
            regime_type = "VOLATILE"
            strength = min(volatility.iloc[-1], 1.0)
        
        return MarketRegime(
            type=regime_type,
            strength=strength,
            confidence=0.8,  # Peut être amélioré avec ML
            volatility=volatility.iloc[-1],
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
        self.price_predictor = GradientBoostingRegressor(
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
        self.current_regime = self._identify_market_regime(df)
        log_debug(f"Régime: {self.current_regime.type} (force: {self.current_regime.strength:.2f})")
        
        # 2. Extraire les features
        features = self._extract_features(df)
        
        # 3. Prédictions ML (si entraîné)
        ml_signal = 0
        ml_confidence = 0.5
        
        if self.is_trained:
            features_scaled = self.feature_scaler.transform([features])
            
            # Prédiction de direction
            price_pred = self.price_predictor.predict_proba(features_scaled)[0]
            ml_signal = price_pred[1] - 0.5  # Centré sur 0
            
            # Évaluation du risque
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
            # Calcul du prix d'entrée et des niveaux
            current_price = df['close'].iloc[-1]
            atr = talib.ATR(df['high'], df['low'], df['close']).iloc[-1]
            
            # Ajuster stop/target selon le régime
            stop_distance = atr * 2
            target_distance = atr * 3
            
            if self.current_regime.type == "VOLATILE":
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
                'regime': self.current_regime.type,
                'reason': self._generate_entry_reason(tech_signals, pattern_signal, ml_signal)
            }
        
        return None
    
    def _analyze_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Analyse technique multi-indicateurs avec scoring"""
        signals = []
        
        # RSI
        rsi = talib.RSI(df['close']).iloc[-1]
        if rsi < 30:
            signals.append({'name': 'RSI_oversold', 'score': 1.0, 'confidence': 0.9})
        elif rsi > 70:
            signals.append({'name': 'RSI_overbought', 'score': -1.0, 'confidence': 0.9})
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'])
        if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
            signals.append({'name': 'MACD_bullish_cross', 'score': 1.0, 'confidence': 0.8})
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'])
        if df['close'].iloc[-1] < lower.iloc[-1]:
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
        
        reasons.append(f"Régime: {self.current_regime.type}")
        
        return " | ".join(reasons)
    
    def should_exit(self, df: pd.DataFrame, position: dict) -> dict:
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