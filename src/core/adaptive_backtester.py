# src/core/adaptive_backtester.py
"""
Backtester adaptatif avec optimisation automatique des paramètres
Utilise Optuna + Ray pour parallélisation
"""

import optuna
from optuna.samplers import TPESampler
import ray
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
from collections import defaultdict

from .backtester import Backtester, BacktestResult
from ..strategies.ai_enhanced_strategy import AIEnhancedStrategy
from .logger import log_info, log_error


class AdaptiveBacktester:
    """
    Backtester qui optimise automatiquement les paramètres
    et s'adapte aux conditions de marché changeantes
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.optimization_history = []
        self.best_params_by_regime = {}
        
        # Initialiser Ray si pas déjà fait
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    def optimize_strategy(self, symbol: str, data: pd.DataFrame, 
                         n_trials: int = 100) -> Dict:
        """
        Optimise les paramètres de la stratégie avec Optuna
        """
        log_info(f"Début optimisation pour {symbol} avec {n_trials} essais...")
        
        def objective(trial):
            # Suggérer les paramètres
            params = {
                # Paramètres ML
                'ml_confidence_threshold': trial.suggest_float('ml_confidence_threshold', 0.5, 0.9),
                'retrain_interval_hours': trial.suggest_int('retrain_interval_hours', 12, 48),
                
                # Risk management
                'base_risk_per_trade': trial.suggest_float('base_risk_per_trade', 0.005, 0.02),
                'max_risk_per_trade': trial.suggest_float('max_risk_per_trade', 0.01, 0.03),
                
                # Indicateurs techniques
                'rsi_period': trial.suggest_int('rsi_period', 7, 21),
                'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 35),
                'rsi_overbought': trial.suggest_int('rsi_overbought', 65, 80),
                
                # Bollinger Bands
                'bb_period': trial.suggest_int('bb_period', 15, 25),
                'bb_std': trial.suggest_float('bb_std', 1.5, 2.5),
                
                # Régime adaptatif
                'regime_lookback_periods': trial.suggest_int('regime_lookback_periods', 50, 200),
                
                # Seuils de décision
                'entry_threshold': trial.suggest_float('entry_threshold', 0.2, 0.5),
                'exit_threshold': trial.suggest_float('exit_threshold', -0.5, -0.2),
            }
            
            # Créer la stratégie avec ces paramètres
            strategy = AIEnhancedStrategy(symbol, params)
            
            # Backtester
            backtester = Backtester(strategy, initial_capital=self.initial_capital)
            result = backtester.run(data)
            
            # Objectif multi-critères
            sharpe = result.sharpe_ratio
            profit_factor = result.profit_factor if result.profit_factor != float('inf') else 10
            max_dd = abs(result.max_drawdown)
            
            # Score composite (à maximiser)
            score = (
                sharpe * 0.4 +  # 40% Sharpe
                min(profit_factor, 3) * 0.3 +  # 30% Profit factor (plafonné)
                (1 - max_dd / 20) * 0.3  # 30% Drawdown (normalisé)
            )
            
            return score
        
        # Créer l'étude Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name=f"{symbol}_optimization"
        )
        
        # Optimiser
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        
        # Résultats
        best_params = study.best_params
        best_score = study.best_value
        
        log_info(f"✅ Optimisation terminée - Score: {best_score:.3f}")
        
        # Sauvegarder l'historique
        self.optimization_history.append({
            'symbol': symbol,
            'timestamp': datetime.now(),
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_trials
        })
        
        return {
            'params': best_params,
            'score': best_score,
            'study': study
        }
    
    @ray.remote
    def _backtest_regime(self, symbol: str, regime_data: pd.DataFrame, 
                        params: Dict) -> BacktestResult:
        """Backtest sur un régime spécifique (pour parallélisation)"""
        strategy = AIEnhancedStrategy(symbol, params)
        backtester = Backtester(strategy)
        return backtester.run(regime_data)
    
    async def adaptive_optimization(self, symbol: str, data: pd.DataFrame,
                                  reoptimize_days: int = 30) -> Dict:
        """
        Optimisation adaptative qui ajuste les paramètres
        selon les régimes de marché
        """
        log_info(f"Démarrage optimisation adaptative pour {symbol}")
        
        # Identifier les régimes dans les données
        regimes = self._identify_regime_periods(data)
        
        # Optimiser pour chaque régime
        regime_results = {}
        
        for regime_type, periods in regimes.items():
            log_info(f"Optimisation pour régime {regime_type}...")
            
            # Combiner les données des périodes du même régime
            regime_data = pd.concat([data.iloc[start:end] for start, end in periods])
            
            if len(regime_data) < 100:
                continue
            
            # Optimiser
            result = self.optimize_strategy(symbol, regime_data, n_trials=50)
            regime_results[regime_type] = result
            
            # Sauvegarder les meilleurs params par régime
            self.best_params_by_regime[regime_type] = result['params']
        
        # Walk-forward analysis
        wf_results = await self._walk_forward_analysis(symbol, data, reoptimize_days)
        
        return {
            'regime_results': regime_results,
            'walk_forward': wf_results,
            'best_params_by_regime': self.best_params_by_regime
        }
    
    def _identify_regime_periods(self, data: pd.DataFrame) -> Dict[str, List[Tuple[int, int]]]:
        """Identifie les périodes de différents régimes de marché"""
        # Utiliser une stratégie simple pour identifier les régimes
        strategy = AIEnhancedStrategy("DUMMY")
        
        regimes = []
        window_size = 100
        
        for i in range(window_size, len(data)):
            window = data.iloc[i-window_size:i]
            regime = strategy._identify_market_regime(window)
            regimes.append(regime.type)
        
        # Grouper les périodes consécutives du même régime
        regime_periods = defaultdict(list)
        current_regime = regimes[0]
        start_idx = window_size
        
        for i, regime in enumerate(regimes[1:], 1):
            if regime != current_regime:
                # Fin du régime actuel
                regime_periods[current_regime].append((start_idx, window_size + i))
                current_regime = regime
                start_idx = window_size + i
        
        # Dernière période
        regime_periods[current_regime].append((start_idx, len(data)))
        
        return dict(regime_periods)
    
    async def _walk_forward_analysis(self, symbol: str, data: pd.DataFrame,
                                   reoptimize_days: int) -> List[Dict]:
        """
        Walk-forward analysis avec ré-optimisation périodique
        """
        results = []
        
        train_size = int(len(data) * 0.7)
        test_size = reoptimize_days * 24 * 4  # 4 périodes de 15min par heure
        
        current_idx = train_size
        
        while current_idx + test_size < len(data):
            # Données d'entraînement
            train_data = data.iloc[current_idx-train_size:current_idx]
            
            # Optimiser sur train
            opt_result = self.optimize_strategy(symbol, train_data, n_trials=30)
            
            # Tester sur out-of-sample
            test_data = data.iloc[current_idx:current_idx+test_size]
            
            strategy = AIEnhancedStrategy(symbol, opt_result['params'])
            backtester = Backtester(strategy)
            test_result = backtester.run(test_data)
            
            results.append({
                'train_end': data.index[current_idx],
                'test_end': data.index[min(current_idx+test_size, len(data)-1)],
                'params': opt_result['params'],
                'train_score': opt_result['score'],
                'test_sharpe': test_result.sharpe_ratio,
                'test_return': test_result.total_return_pct,
                'test_drawdown': test_result.max_drawdown
            })
            
            # Avancer
            current_idx += test_size
        
        return results
    
    def save_optimization_results(self, filepath: str):
        """Sauvegarde les résultats d'optimisation"""
        results = {
            'optimization_history': self.optimization_history,
            'best_params_by_regime': self.best_params_by_regime,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        log_info(f"Résultats d'optimisation sauvegardés dans {filepath}")