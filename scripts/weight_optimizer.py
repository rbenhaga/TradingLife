"""
Module d'optimisation des poids pour les stratégies multi-signaux
Utilise Optuna pour trouver les meilleurs poids d'indicateurs
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from src.core.logger import log_info, log_error, log_debug
from src.core.backtester import Backtester, BacktestResult
from src.strategies.strategy import MultiSignalStrategy
from src.exchanges.exchange_connector import ExchangeConnector


@dataclass
class OptimizationResult:
    """Résultat d'une optimisation"""
    symbol: str
    best_weights: Dict[str, float]
    sharpe_ratio: float
    total_return: float
    profit_factor: float
    win_rate: float
    max_drawdown: float
    best_value: float
    n_trials: int
    optimization_time: float


class WeightOptimizer:
    """
    Optimiseur de poids pour les indicateurs techniques
    Trouve la combinaison optimale de poids pour maximiser une métrique
    """
    
    def __init__(self, exchange: Optional[ExchangeConnector] = None):
        """
        Initialise l'optimiseur
        
        Args:
            exchange: Connecteur d'exchange pour récupérer les données
        """
        self.exchange = exchange
        self.optimization_metric = 'sharpe_ratio'  # Métrique par défaut
        self.optimization_history = []
        
        # Limites des poids
        self.weight_bounds = {
            'rsi': (0.05, 0.30),
            'bollinger': (0.05, 0.30),
            'macd': (0.05, 0.25),
            'volume': (0.05, 0.25),
            'ma_cross': (0.05, 0.20),
            'momentum': (0.05, 0.20),
            'volatility': (0.05, 0.20)
        }
        
        log_info("WeightOptimizer initialisé")
    
    async def _get_historical_data(self, symbol: str, start_date: datetime, 
                                  end_date: datetime) -> pd.DataFrame:
        """
        Récupère les données historiques
        
        Args:
            symbol: Symbole à récupérer
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame avec les données OHLCV
        """
        if not self.exchange:
            raise ValueError("Exchange connector non configuré")
        
        # Calculer le nombre de périodes nécessaires
        delta = end_date - start_date
        periods = int(delta.total_seconds() / (15 * 60))  # 15 minutes
        
        # Récupérer les données
        ohlcv = await self.exchange.get_ohlcv(symbol, '15m', limit=min(periods, 1000))
        
        # Convertir en DataFrame
        columns = pd.Index(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = pd.DataFrame(ohlcv, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filtrer par dates
        df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        
        return df
    
    def _objective(self, trial: optuna.Trial, symbol: str, 
                   data: pd.DataFrame, initial_capital: float) -> float:
        """
        Fonction objectif pour Optuna
        
        Args:
            trial: Essai Optuna
            symbol: Symbole tradé
            data: Données historiques
            initial_capital: Capital initial
            
        Returns:
            Valeur de la métrique à optimiser
        """
        # Suggérer des poids
        weights = {}
        remaining = 1.0
        
        indicators = list(self.weight_bounds.keys())
        
        # Générer des poids qui somment à 1
        for i, indicator in enumerate(indicators[:-1]):
            min_w, max_w = self.weight_bounds[indicator]
            # Ajuster les bornes pour respecter la somme = 1
            max_possible = min(max_w, remaining - sum(self.weight_bounds[ind][0] for ind in indicators[i+1:]))
            min_possible = max(min_w, remaining - sum(self.weight_bounds[ind][1] for ind in indicators[i+1:]))
            
            if max_possible >= min_possible:
                weights[indicator] = trial.suggest_float(
                    f'weight_{indicator}',
                    min_possible,
                    max_possible
                )
                remaining -= weights[indicator]
            else:
                # Si impossible, utiliser le minimum
                weights[indicator] = min_w
                remaining -= min_w
        
        # Le dernier prend ce qui reste
        weights[indicators[-1]] = remaining
        
        # Créer la stratégie avec ces poids
        strategy = MultiSignalStrategy(symbol, score_weights=weights)
        
        # Backtester
        backtester = Backtester(strategy, initial_capital=initial_capital)
        result = backtester.run(data)
        
        # Retourner la métrique demandée
        if self.optimization_metric == 'sharpe_ratio':
            return result.sharpe_ratio
        elif self.optimization_metric == 'profit_factor':
            return result.profit_factor if result.profit_factor != float('inf') else 10.0
        elif self.optimization_metric == 'total_return':
            return result.total_return_pct
        elif self.optimization_metric == 'calmar_ratio':
            return result.metrics.get('calmar_ratio', 0)
        else:
            return result.sharpe_ratio
    
    def optimize(self, symbol: str, start_date: datetime, end_date: datetime,
                n_trials: int = 100, initial_capital: float = 10000) -> OptimizationResult:
        """
        Optimise les poids pour un symbole
        
        Args:
            symbol: Symbole à optimiser
            start_date: Date de début des données
            end_date: Date de fin des données
            n_trials: Nombre d'essais Optuna
            initial_capital: Capital initial pour le backtest
            
        Returns:
            Résultat de l'optimisation
        """
        log_info(f"Début optimisation {symbol} - {n_trials} essais")
        start_time = datetime.now()
        
        # Récupérer les données
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(
            self._get_historical_data(symbol, start_date, end_date)
        )
        
        if len(data) < 100:
            raise ValueError(f"Pas assez de données pour {symbol}: {len(data)} périodes")
        
        # Créer l'étude Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name=f'{symbol}_weight_optimization'
        )
        
        # Optimiser
        study.optimize(
            lambda trial: self._objective(trial, symbol, data, initial_capital),
            n_trials=n_trials,
            n_jobs=1  # Éviter les problèmes de parallélisation sur Windows
        )
        
        # Récupérer les meilleurs paramètres
        best_weights = {}
        for param, value in study.best_params.items():
            if param.startswith('weight_'):
                indicator = param.replace('weight_', '')
                best_weights[indicator] = value
        
        # Compléter avec le dernier poids
        indicators = list(self.weight_bounds.keys())
        last_indicator = indicators[-1]
        if last_indicator not in best_weights:
            best_weights[last_indicator] = 1.0 - sum(best_weights.values())
        
        # Normaliser pour être sûr que la somme = 1
        total = sum(best_weights.values())
        best_weights = {k: v/total for k, v in best_weights.items()}
        
        # Faire un backtest final avec les meilleurs poids
        strategy = MultiSignalStrategy(symbol, score_weights=best_weights)
        backtester = Backtester(strategy, initial_capital=initial_capital)
        final_result = backtester.run(data)
        
        # Calculer le temps d'optimisation
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Créer le résultat
        result = OptimizationResult(
            symbol=symbol,
            best_weights=best_weights,
            sharpe_ratio=final_result.sharpe_ratio,
            total_return=final_result.total_return_pct,
            profit_factor=final_result.profit_factor if final_result.profit_factor != float('inf') else 10.0,
            win_rate=final_result.win_rate,
            max_drawdown=final_result.max_drawdown,
            best_value=study.best_value,
            n_trials=n_trials,
            optimization_time=optimization_time
        )
        
        # Ajouter à l'historique
        self.optimization_history.append(result)
        
        log_info(f"Optimisation terminée - Best {self.optimization_metric}: {study.best_value:.3f}")
        self._log_results(result)
        
        return result
    
    def optimize_multiple_pairs(self, symbols: List[str], n_trials: int = 100,
                               days_back: int = 30, initial_capital: float = 10000) -> Dict[str, OptimizationResult]:
        """
        Optimise plusieurs paires en parallèle
        
        Args:
            symbols: Liste des symboles à optimiser
            n_trials: Nombre d'essais par symbole
            days_back: Nombre de jours d'historique
            initial_capital: Capital initial
            
        Returns:
            Dict des résultats par symbole
        """
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Utiliser multiprocessing pour paralléliser
        with ProcessPoolExecutor(max_workers=min(len(symbols), mp.cpu_count() - 1)) as executor:
            # Soumettre les tâches
            futures = {}
            for symbol in symbols:
                future = executor.submit(
                    self._optimize_single_wrapper,
                    symbol, start_date, end_date, n_trials, initial_capital
                )
                futures[future] = symbol
            
            # Récupérer les résultats
            for future in futures:
                symbol = futures[future]
                try:
                    result = future.result()
                    results[symbol] = result
                    log_info(f"✅ {symbol} optimisé avec succès")
                except Exception as e:
                    log_error(f"❌ Erreur optimisation {symbol}: {str(e)}")
        
        return results
    
    def _optimize_single_wrapper(self, symbol: str, start_date: datetime, 
                                end_date: datetime, n_trials: int, 
                                initial_capital: float) -> OptimizationResult:
        """Wrapper pour l'optimisation dans un processus séparé"""
        # Recréer l'exchange connector dans le nouveau processus
        self.exchange = ExchangeConnector(
            exchange_name='binance',
            testnet=True
        )
        
        return self.optimize(symbol, start_date, end_date, n_trials, initial_capital)
    
    def _log_results(self, result: OptimizationResult):
        """Affiche les résultats de l'optimisation"""
        log_info("📊 Résultats de l'optimisation:")
        log_info(f"  - Symbole: {result.symbol}")
        log_info(f"  - Sharpe Ratio: {result.sharpe_ratio:.3f}")
        log_info(f"  - Return Total: {result.total_return:.2f}%")
        log_info(f"  - Win Rate: {result.win_rate:.1f}%")
        log_info(f"  - Max Drawdown: {result.max_drawdown:.2f}%")
        log_info(f"  - Temps: {result.optimization_time:.1f}s")
        
        log_info("⚖️ Poids optimaux:")
        for indicator, weight in sorted(result.best_weights.items(), key=lambda x: x[1], reverse=True):
            log_info(f"  - {indicator}: {weight:.1%}")
    
    def get_best_weights(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Récupère les meilleurs poids pour un symbole
        
        Args:
            symbol: Symbole recherché
            
        Returns:
            Dict des poids ou None si pas trouvé
        """
        for result in reversed(self.optimization_history):
            if result.symbol == symbol:
                return result.best_weights
        return None
    
    def save_results(self, filepath: str = 'optimization_results.json'):
        """Sauvegarde les résultats d'optimisation"""
        import json
        
        data = {
            'optimization_date': datetime.now().isoformat(),
            'metric': self.optimization_metric,
            'results': []
        }
        
        for result in self.optimization_history:
            data['results'].append({
                'symbol': result.symbol,
                'best_weights': result.best_weights,
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': result.total_return,
                'profit_factor': result.profit_factor,
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'n_trials': result.n_trials,
                'optimization_time': result.optimization_time
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        log_info(f"Résultats sauvegardés dans {filepath}")