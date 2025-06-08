#!/usr/bin/env python
"""
Script d'optimisation des poids des indicateurs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

from src.exchanges.exchange_connector import ExchangeConnector
from src.core.weight_optimizer import WeightOptimizer

def main():
    parser = argparse.ArgumentParser(
        description="Optimise les poids des indicateurs techniques"
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='BTC/USDT',
        help='Paire à optimiser (défaut: BTC/USDT)'
    )
    
    parser.add_argument(
        '--symbols', '-m',
        type=str,
        nargs='+',
        help='Optimiser plusieurs paires (ex: BTC/USDT ETH/USDT)'
    )
    
    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=100,
        help='Nombre d'essais (défaut: 100)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Nombre de jours d'historique (défaut: 30)'
    )
    
    parser.add_argument(
        '--metric', '-me',
        type=str,
        default='sharpe_ratio',
        choices=['sharpe_ratio', 'profit_factor', 'total_return', 'calmar_ratio'],
        help='Métrique à optimiser (défaut: sharpe_ratio)'
    )
    
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=10000,
        help='Capital initial pour le backtest (défaut: 10000)'
    )
    
    parser.add_argument(
        '--testnet',
        action='store_true',
        help='Utiliser le testnet'
    )
    
    parser.add_argument(
        '--save-config',
        action='store_true',
        help='Sauvegarder les poids optimaux dans la config'
    )
    
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('OptimizeWeights')
    
    # Charger les variables d'environnement
    load_dotenv()
    
    print("\n" + "="*60)
    print("🧬 OPTIMISATION DES POIDS DES INDICATEURS")
    print("="*60)
    
    try:
        # Créer l'exchange connector
        logger.info("Connexion à l'exchange...")
        exchange = ExchangeConnector(
            exchange_name='binance',
            testnet=args.testnet
        )
        
        # Créer l'optimiseur
        optimizer = WeightOptimizer(exchange)
        optimizer.optimization_metric = args.metric
        
        # Calculer les dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Optimiser une ou plusieurs paires
        if args.symbols:
            # Optimisation multiple
            logger.info(f"Optimisation de {len(args.symbols)} paires...")
            results = optimizer.optimize_multiple_pairs(
                symbols=args.symbols,
                n_trials=args.trials
            )
            
            # Sauvegarder la config si demandé
            if args.save_config:
                save_optimized_config(results)
                
        else:
            # Optimisation simple
            logger.info(f"Optimisation de {args.symbol}...")
            result = optimizer.optimize(
                symbol=args.symbol,
                start_date=start_date,
                end_date=end_date,
                n_trials=args.trials,
                initial_capital=args.capital
            )
            
            # Sauvegarder la config si demandé
            if args.save_config:
                save_optimized_config({args.symbol: result})
        
        print("\n✅ Optimisation terminée avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'optimisation: {e}")
        raise

def save_optimized_config(results):
    """Sauvegarde les poids optimaux dans un fichier de configuration"""
    config = {
        'optimized_weights': {},
        'optimization_date': datetime.now().isoformat(),
        'performance_summary': {}
    }
    
    for symbol, result in results.items():
        config['optimized_weights'][symbol] = result.best_weights
        config['performance_summary'][symbol] = {
            'sharpe_ratio': result.sharpe_ratio,
            'total_return': result.total_return,
            'profit_factor': result.profit_factor,
            'win_rate': result.win_rate,
            'max_drawdown': result.max_drawdown
        }
    
    # Sauvegarder dans un fichier
    filename = 'config/optimized_config.json'
    Path('config').mkdir(exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Configuration optimale sauvegardée dans {filename}")

if __name__ == "__main__":
    main()
