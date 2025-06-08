"""
Module de backtesting pour les stratégies de trading
Implémentation complète avec vectorisation et optimisation Optuna
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import optuna
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Optional third party libraries
try:
    import vectorbt as vbt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    vbt = None

try:
    import ray  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ray = None

from .logger import log_info, log_error, log_debug
from .weighted_score_engine import WeightedScoreEngine, TradingScore
from ..strategies.strategy import Strategy, MultiSignalStrategy

@dataclass
class BacktestResult:
    """Résultats du backtest"""
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: float
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]

class Backtester:
    """
    Backtester vectorisé pour tester les stratégies sur données historiques
    """
    
    def __init__(self, strategy: Strategy, initial_capital: float = 10000,
                 commission: float = 0.001, slippage: float = 0.0005,
                 engine: str = "custom"):
        """
        Initialise le backtester
        
        Args:
            strategy: Stratégie à tester
            initial_capital: Capital initial
            commission: Commission par trade (0.1% par défaut)
            slippage: Slippage estimé (0.05% par défaut)
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.engine = engine
        
        # État du backtest
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = [initial_capital]
        
        log_info(
            f"Backtester initialisé - Capital: {initial_capital}, Commission: {commission*100}% | Engine: {self.engine}"
        )
    
    def run(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        engine: Optional[str] = None,
    ) -> BacktestResult:
        """
        Exécute le backtest sur les données historiques
        
        Args:
            data: DataFrame avec colonnes OHLCV + indicateurs
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
        
        Returns:
            BacktestResult avec toutes les métriques
        """
        # Filtrer les données par date si nécessaire
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        eng = engine or self.engine
        log_info(f"Début du backtest sur {len(data)} périodes (engine={eng})")

        # Vectorisation: Précalculer tous les signaux
        signals = self._generate_signals_vectorized(data)

        if eng == "vectorbt":
            trades_df, pf = self._execute_trades_vectorbt(data, signals)
            result = self._calculate_metrics_vectorbt(pf)
        else:
            trades_df = self._execute_trades_vectorized(data, signals)
            result = self._calculate_metrics(trades_df)

        log_info(
            f"Backtest terminé - Return: {result.total_return_pct:.2f}%, Sharpe: {result.sharpe_ratio:.2f}"
        )
        
        return result
    
    def _generate_signals_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère tous les signaux de manière vectorisée
        
        Args:
            data: DataFrame avec les données
        
        Returns:
            DataFrame avec les signaux
        """
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        signals['signal'] = 0
        signals['position'] = 0
        
        # Pour MultiSignalStrategy, utiliser le score engine
        if isinstance(self.strategy, MultiSignalStrategy):
            # Calculer les scores pour chaque ligne
            scores = []
            for i in range(len(data)):
                if i < 50:  # Pas assez de données
                    scores.append(0)
                    continue
                
                # Fenêtre glissante
                window_data = data.iloc[max(0, i-100):i+1]
                
                # Analyser avec le score engine
                score_signals = self.strategy.score_engine.analyze_indicators(window_data)
                trading_score = self.strategy.score_engine.calculate_score(score_signals)
                
                scores.append(trading_score.total_score)
            
            signals['score'] = scores
            
            # Générer les signaux basés sur les scores
            signals.loc[signals['score'] > 0.5, 'signal'] = 1
            signals.loc[signals['score'] < -0.5, 'signal'] = -1
        else:
            # Pour les autres stratégies, utiliser une approche simple
            # RSI
            if 'rsi' in data.columns:
                signals.loc[data['rsi'] < 30, 'signal'] = 1
                signals.loc[data['rsi'] > 70, 'signal'] = -1
        
        # Calculer les positions (éviter les signaux répétés)
        signals['position'] = signals['signal'].diff().fillna(0)
        signals.loc[signals['position'] > 0, 'position'] = 1  # Entrée long
        signals.loc[signals['position'] < 0, 'position'] = -1  # Sortie/short
        
        return signals
    
    def _execute_trades_vectorized(self, data: pd.DataFrame,
                                  signals: pd.DataFrame) -> pd.DataFrame:
        """
        Exécute les trades de manière vectorisée
        
        Args:
            data: Données de marché
            signals: Signaux de trading
        
        Returns:
            DataFrame des trades exécutés
        """
        trades = []
        position = None
        capital = self.initial_capital
        equity_curve = [capital]
        
        for i in range(len(signals)):
            current_signal = signals.iloc[i]
            
            # Entrée en position
            if current_signal['position'] == 1 and position is None:
                entry_price = current_signal['price'] * (1 + self.slippage)
                commission_cost = capital * self.commission
                position_size = (capital - commission_cost) / entry_price
                
                position = {
                    'entry_date': signals.index[i],
                    'entry_price': entry_price,
                    'size': position_size,
                    'commission_in': commission_cost
                }
            
            # Sortie de position
            elif current_signal['position'] == -1 and position is not None:
                exit_price = current_signal['price'] * (1 - self.slippage)
                position_value = position['size'] * exit_price
                commission_cost = position_value * self.commission
                
                # Calculer le PnL
                pnl = position_value - (position['size'] * position['entry_price'])
                net_pnl = pnl - position['commission_in'] - commission_cost
                
                # Enregistrer le trade
                trade = {
                    'entry_date': position['entry_date'],
                    'exit_date': signals.index[i],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'net_pnl': net_pnl,
                    'return_pct': (net_pnl / capital) * 100,
                    'duration': (signals.index[i] - position['entry_date']).total_seconds() / 3600
                }
                trades.append(trade)
                
                # Mettre à jour le capital
                capital += net_pnl
                position = None
            
            # Mettre à jour l'equity curve
            if position is not None:
                # Position ouverte - calculer la valeur mark-to-market
                current_value = position['size'] * current_signal['price']
                unrealized_pnl = current_value - (position['size'] * position['entry_price'])
                equity_curve.append(capital + unrealized_pnl)
            else:
                equity_curve.append(capital)
        
        # Convertir en DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        self.equity_curve = pd.Series(equity_curve, index=signals.index)

        return trades_df

    def _execute_trades_vectorbt(
        self, data: pd.DataFrame, signals: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Any]:
        """Execute trades using vectorbt portfolio"""
        if vbt is None:
            raise ImportError("vectorbt library is required for this engine")

        price = data["close"]
        entries = signals["position"] == 1
        exits = signals["position"] == -1

        pf = vbt.Portfolio.from_signals(
            price,
            entries,
            exits,
            init_cash=self.initial_capital,
            fees=self.commission,
            slippage=self.slippage,
        )

        self.equity_curve = pf.value()
        trades_df = pf.trades.records_readable
        return trades_df, pf
    
    def _calculate_metrics(self, trades_df: pd.DataFrame) -> BacktestResult:
        """
        Calcule toutes les métriques du backtest
        
        Args:
            trades_df: DataFrame des trades
        
        Returns:
            BacktestResult avec toutes les métriques
        """
        final_capital = self.equity_curve.iloc[-1]
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Métriques de base
        total_trades = len(trades_df)
        if total_trades == 0:
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                total_return_pct=total_return_pct,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                max_drawdown_duration=0,
                win_rate=0,
                profit_factor=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0,
                avg_loss=0,
                best_trade=0,
                worst_trade=0,
                avg_trade_duration=0,
                equity_curve=self.equity_curve,
                trades=trades_df,
                metrics={}
            )
        
        # Trades gagnants/perdants
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        losing_trades = trades_df[trades_df['net_pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit factor
        total_wins = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Moyennes
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Meilleur/pire trade
        best_trade = trades_df['net_pnl'].max() if total_trades > 0 else 0
        worst_trade = trades_df['net_pnl'].min() if total_trades > 0 else 0
        
        # Durée moyenne
        avg_trade_duration = trades_df['duration'].mean() if total_trades > 0 else 0
        
        # Calcul du Sharpe Ratio
        returns = self.equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # Annualisé
        else:
            sharpe_ratio = 0
        
        # Calcul du Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252 * 24)
        else:
            sortino_ratio = 0
        
        # Calcul du Maximum Drawdown
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Durée du max drawdown
        drawdown_duration = 0
        if max_drawdown < 0:
            dd_start = drawdown[drawdown == drawdown.min()].index[0]
            dd_data = drawdown[drawdown.index >= dd_start]
            recovery_dates = dd_data[dd_data >= 0].index
            if len(recovery_dates) > 0:
                dd_end = recovery_dates[0]
                drawdown_duration = (dd_end - dd_start).days
            else:
                drawdown_duration = (drawdown.index[-1] - dd_start).days
        
        # Métriques additionnelles
        metrics = {
            'calmar_ratio': abs(total_return_pct / max_drawdown) if max_drawdown != 0 else 0,
            'avg_trade_return': trades_df['return_pct'].mean() if total_trades > 0 else 0,
            'trade_return_std': trades_df['return_pct'].std() if total_trades > 0 else 0,
            'max_consecutive_wins': self._max_consecutive(winning_trades),
            'max_consecutive_losses': self._max_consecutive(losing_trades),
            'exposure_time': (total_trades * avg_trade_duration) / (len(self.equity_curve) * 24) if len(self.equity_curve) > 0 else 0
        }
        
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=drawdown_duration,
            win_rate=win_rate * 100,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration=avg_trade_duration,
            equity_curve=self.equity_curve,
            trades=trades_df,
            metrics=metrics
        )

    def _calculate_metrics_vectorbt(self, pf: Any) -> BacktestResult:
        """Calculate metrics using vectorbt Portfolio object"""
        equity_curve = pf.value()

        final_capital = equity_curve.iloc[-1]
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = (
            (returns.mean() / returns.std()) * np.sqrt(252 * 24)
            if len(returns) > 0 and returns.std() > 0
            else 0
        )
        downside = returns[returns < 0]
        sortino_ratio = (
            (returns.mean() / downside.std()) * np.sqrt(252 * 24)
            if len(downside) > 0 and downside.std() > 0
            else 0
        )

        max_drawdown = ((equity_curve / equity_curve.cummax()) - 1).min() * 100

        trades = pf.trades
        pnl = trades.pnl
        total_trades = len(pnl)
        winning_trades = (pnl > 0).sum()
        losing_trades = (pnl <= 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (
            pnl[pnl > 0].sum() / abs(pnl[pnl <= 0].sum())
            if losing_trades > 0
            else float("inf")
        )
        avg_win = pnl[pnl > 0].mean() if winning_trades > 0 else 0
        avg_loss = pnl[pnl <= 0].mean() if losing_trades > 0 else 0
        best_trade = pnl.max() if total_trades > 0 else 0
        worst_trade = pnl.min() if total_trades > 0 else 0
        avg_trade_duration = trades.duration.mean() if total_trades > 0 else 0

        metrics = {
            "calmar_ratio": abs(total_return_pct / max_drawdown)
            if max_drawdown != 0
            else 0
        }

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=0,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=int(winning_trades),
            losing_trades=int(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration=avg_trade_duration,
            equity_curve=equity_curve,
            trades=pf.trades.records_readable,
            metrics=metrics,
        )
    
    def _max_consecutive(self, trades_df: pd.DataFrame) -> int:
        """Calcule le nombre maximum de trades consécutifs"""
        if len(trades_df) == 0:
            return 0
        
        # Utiliser les dates pour déterminer la séquence
        consecutive = 1
        max_consecutive = 1
        
        for i in range(1, len(trades_df)):
            if trades_df.index[i] == trades_df.index[i-1] + 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1
        
        return max_consecutive

    def run_parallel(self, data: pd.DataFrame, param_grid: List[Dict]) -> List[BacktestResult]:
        """Execute several backtests in parallel using Ray"""
        if ray is None:
            raise ImportError("ray library is required for parallel execution")

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        @ray.remote
        def _run(params: Dict) -> BacktestResult:
            strategy = self.strategy.__class__(**params)
            bt = Backtester(
                strategy,
                initial_capital=self.initial_capital,
                commission=self.commission,
                slippage=self.slippage,
                engine=self.engine,
            )
            return bt.run(data, engine=self.engine)

        futures = [_run.remote(params) for params in param_grid]
        results = ray.get(futures)
        ray.shutdown()
        return results
    
    def optimize_strategy(self, data: pd.DataFrame, 
                         n_trials: int = 100,
                         metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimise les paramètres de la stratégie avec Optuna
        
        Args:
            data: Données historiques
            n_trials: Nombre d'essais
            metric: Métrique à optimiser
        
        Returns:
            Dict avec les meilleurs paramètres et résultats
        """
        log_info(f"Début de l'optimisation Optuna - {n_trials} trials, métrique: {metric}")
        
        def objective(trial):
            # Suggérer des paramètres
            if isinstance(self.strategy, MultiSignalStrategy):
                # Optimiser les poids du score engine
                weights = {
                    'rsi': trial.suggest_float('weight_rsi', 0.05, 0.30),
                    'bollinger': trial.suggest_float('weight_bollinger', 0.05, 0.30),
                    'macd': trial.suggest_float('weight_macd', 0.05, 0.30),
                    'volume': trial.suggest_float('weight_volume', 0.05, 0.25),
                    'ma_cross': trial.suggest_float('weight_ma_cross', 0.05, 0.20),
                    'momentum': trial.suggest_float('weight_momentum', 0.05, 0.20),
                    'volatility': trial.suggest_float('weight_volatility', 0.05, 0.20)
                }
                
                # Mettre à jour les poids
                self.strategy.score_engine.update_weights(weights)
                
                # Suggérer les seuils
                self.strategy.entry_threshold = trial.suggest_float('entry_threshold', 0.3, 0.8)
                self.strategy.exit_threshold = trial.suggest_float('exit_threshold', -0.8, -0.3)
            
            # Exécuter le backtest
            result = self.run(data)
            
            # Retourner la métrique à optimiser
            if metric == 'sharpe_ratio':
                return result.sharpe_ratio
            elif metric == 'profit_factor':
                return result.profit_factor if result.profit_factor != float('inf') else 10
            elif metric == 'total_return':
                return result.total_return_pct
            elif metric == 'calmar_ratio':
                return result.metrics['calmar_ratio']
            else:
                return result.sharpe_ratio
        
        # Créer l'étude Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimiser
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        # Récupérer les meilleurs paramètres
        best_params = study.best_params
        best_value = study.best_value
        
        log_info(f"Optimisation terminée - Meilleure {metric}: {best_value:.3f}")
        log_info(f"Meilleurs paramètres: {best_params}")
        
        # Exécuter un dernier backtest avec les meilleurs paramètres
        if isinstance(self.strategy, MultiSignalStrategy):
            weights = {k.replace('weight_', ''): v for k, v in best_params.items() if k.startswith('weight_')}
            self.strategy.score_engine.update_weights(weights)
            
            if 'entry_threshold' in best_params:
                self.strategy.entry_threshold = best_params['entry_threshold']
            if 'exit_threshold' in best_params:
                self.strategy.exit_threshold = best_params['exit_threshold']
        
        final_result = self.run(data)
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'final_result': final_result,
            'optimization_history': study.trials_dataframe()
        }

class WalkForwardOptimizer:
    """
    Optimiseur Walk-Forward pour validation robuste
    """
    
    def __init__(self, strategy_class, n_splits: int = 5):
        """
        Initialise l'optimiseur walk-forward
        
        Args:
            strategy_class: Classe de stratégie à optimiser
            n_splits: Nombre de splits pour la validation
        """
        self.strategy_class = strategy_class
        self.n_splits = n_splits
    
    def run(self, data: pd.DataFrame, 
            train_pct: float = 0.7,
            n_trials: int = 50) -> pd.DataFrame:
        """
        Exécute l'optimisation walk-forward
        
        Args:
            data: Données complètes
            train_pct: Pourcentage pour l'entraînement
            n_trials: Trials par split
        
        Returns:
            DataFrame avec les résultats de chaque split
        """
        results = []
        split_size = len(data) // self.n_splits
        
        for i in range(self.n_splits - 1):
            # Définir les périodes
            start_idx = i * split_size
            end_idx = (i + 2) * split_size
            
            train_end_idx = start_idx + int(split_size * train_pct)
            
            # Séparer train/test
            train_data = data.iloc[start_idx:train_end_idx]
            test_data = data.iloc[train_end_idx:end_idx]
            
            # Optimiser sur train
            strategy = self.strategy_class('BTC/USDT')  # Exemple
            backtester = Backtester(strategy)
            opt_result = backtester.optimize_strategy(train_data, n_trials=n_trials)
            
            # Tester sur test
            test_result = backtester.run(test_data)
            
            results.append({
                'split': i,
                'train_sharpe': opt_result['best_value'],
                'test_sharpe': test_result.sharpe_ratio,
                'test_return': test_result.total_return_pct,
                'test_drawdown': test_result.max_drawdown,
                'best_params': opt_result['best_params']
            })
        
        return pd.DataFrame(results)
