#!/usr/bin/env python3
"""
Script d'optimisation des poids des indicateurs
"""

import argparse
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import ccxt

from src.core.weight_optimizer import WeightOptimizer
from src.exchanges.exchange_connector import ExchangeConnector
from src.core.backtester import Backtester
from src.strategies.strategy import MultiSignalStrategy
import pandas as pd
import numpy as np
import typing as t
import optuna
from optuna.samplers import TPESampler

def load_testnet_data(symbol='BTC/USDT', timeframe='1h', days=30):
    """Charge les données depuis l'API publique (pas besoin de clés)"""
    print(f"📊 Chargement des données {symbol} depuis l'API publique...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        }
    })
    
    # Utiliser l'API publique (pas besoin de clés)
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    
    all_data = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"Erreur: {e}")
            break
    
    df = pd.DataFrame(all_data, columns=pd.Index(['timestamp', 'open', 'high', 'low', 'close', 'volume']))
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ {len(df)} périodes chargées")
    return df

async def run_optimization_offline(symbol, days, n_trials, metric, initial_capital):
    """Optimisation avec données téléchargées"""
    
    # Charger les données depuis l'API publique
    data = load_testnet_data(symbol, '1h', days)
    
    if len(data) < 100:
        raise ValueError(f"Pas assez de données: {len(data)} périodes")
    
    print(f"\n🔧 Optimisation sur {len(data)} périodes...")
    
    # Fonction objective pour Optuna
    def objective(trial):
        # Suggérer les poids
        weights = {
            'rsi': trial.suggest_float('weight_rsi', 0.05, 0.30),
            'bollinger': trial.suggest_float('weight_bollinger', 0.05, 0.30),
            'macd': trial.suggest_float('weight_macd', 0.05, 0.25),
            'volume': trial.suggest_float('weight_volume', 0.05, 0.25),
            'ma_cross': trial.suggest_float('weight_ma_cross', 0.05, 0.20),
            'momentum': trial.suggest_float('weight_momentum', 0.05, 0.20)
        }
        
        # Normaliser les poids
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Créer la stratégie avec ces poids
        strategy = MultiSignalStrategy(symbol, score_weights=weights)
        
        # Suggérer les seuils
        strategy.entry_threshold = trial.suggest_float('entry_threshold', 0.3, 0.7)
        strategy.exit_threshold = trial.suggest_float('exit_threshold', -0.7, -0.3)
        
        # Backtester
        backtester = Backtester(strategy, initial_capital=initial_capital)
        result = backtester.run(data)
        
        # Retourner la métrique à optimiser
        if metric == 'sharpe_ratio':
            return result.sharpe_ratio if not np.isnan(result.sharpe_ratio) else -10
        elif metric == 'total_return':
            return result.total_return_pct
        elif metric == 'profit_factor':
            return result.profit_factor if result.profit_factor != float('inf') else 10
        else:
            return result.win_rate
    
    # Créer l'étude Optuna
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    # Progress callback
    def callback(study, trial):
        if trial.number % 10 == 0:
            print(f"  Trial {trial.number}: {metric}={trial.value:.3f}")
    
    # Optimiser
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    # Résultats
    best_trial = study.best_trial
    print(f"\n✅ Optimisation terminée!")
    print(f"Meilleur {metric}: {best_trial.value:.3f}")
    
    # Extraire les poids
    best_weights = {}
    for param, value in best_trial.params.items():
        if param.startswith('weight_'):
            indicator = param.replace('weight_', '')
            best_weights[indicator] = value
    
    # Normaliser
    total = sum(best_weights.values())
    best_weights = {k: v/total for k, v in best_weights.items()}
    
    return {
        'best_weights': best_weights,
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'optimization_history': study.trials_dataframe()
    }

async def main():
    parser = argparse.ArgumentParser(description='Optimiser les poids des indicateurs')
    parser.add_argument('--symbol', nargs='+', default=['BTC/USDT'], help='Symboles à optimiser')
    parser.add_argument('--days', type=int, default=30, help='Jours d\'historique')
    parser.add_argument('--trials', type=int, default=50, help='Nombre d\'essais Optuna')
    parser.add_argument('--metric', choices=['sharpe_ratio', 'total_return', 'profit_factor', 'win_rate'],
                       default='sharpe_ratio', help='Métrique à optimiser')
    parser.add_argument('--capital', type=float, default=10000, help='Capital initial')
    parser.add_argument('--save-config', action='store_true', help='Sauvegarder la configuration')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🧬 OPTIMISATION DES POIDS (Mode Offline)")
    print("="*60)
    print(f"📊 Métrique: {args.metric}")
    print(f"🔄 Essais: {args.trials}")
    print(f"📅 Historique: {args.days} jours")
    print(f"💰 Capital: {args.capital} USDT")
    print("="*60 + "\n")
    
    for symbol in args.symbol:
        try:
            print(f"\n🎯 Optimisation de {symbol}...")
            result = await run_optimization_offline(
                symbol, args.days, args.trials, args.metric, args.capital
            )
            
            # Afficher les résultats
            print(f"\n📊 Meilleurs poids pour {symbol}:")
            print("-" * 40)
            for indicator, weight in sorted(result['best_weights'].items(), 
                                         key=lambda x: x[1], reverse=True):
                print(f"  {indicator:<12} : {weight:>6.1%}")
            
            # Autres paramètres
            print(f"\n📈 Autres paramètres optimaux:")
            print(f"  Seuil entrée  : {result['best_params'].get('entry_threshold', 0.5):.2f}")
            print(f"  Seuil sortie  : {result['best_params'].get('exit_threshold', -0.3):.2f}")
            
            # Sauvegarder si demandé
            if args.save_config:
                config = {
                    'strategy': {
                        'weights': result['best_weights'],
                        'entry_threshold': result['best_params'].get('entry_threshold', 0.5),
                        'exit_threshold': result['best_params'].get('exit_threshold', -0.3)
                    },
                    'optimization': {
                        'date': datetime.now().isoformat(),
                        'metric': args.metric,
                        'value': result['best_value'],
                        'trials': args.trials
                    }
                }
                
                filename = f"config/optimized_{symbol.replace('/', '_')}.json"
                import json
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"\n💾 Configuration sauvegardée: {filename}")
                
        except Exception as e:
            print(f"\n❌ Erreur pour {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())