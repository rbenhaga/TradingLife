#!/usr/bin/env python
"""
Script d'optimisation des poids des indicateurs
Version améliorée avec gestion d'erreurs et affichage détaillé
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import asyncio
import os

from src.exchanges.exchange_connector import ExchangeConnector
from src.core.weight_optimizer import WeightOptimizer

async def main():
    parser = argparse.ArgumentParser(
        description="Optimise les poids des indicateurs techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Optimiser BTC/USDT avec 100 essais
  python optimize_weights.py --symbol BTC/USDT --trials 100
  
  # Optimiser plusieurs paires
  python optimize_weights.py --symbols BTC/USDT ETH/USDT BNB/USDT --trials 50
  
  # Optimiser avec métrique personnalisée et sauvegarder
  python optimize_weights.py --symbol BTC/USDT --metric profit_factor --save-config
        """
    )
    
    # Arguments...
    parser.add_argument('--symbol', type=str, help='Symbole à optimiser (ex: BTC/USDT)')
    parser.add_argument('--symbols', nargs='+', help='Liste de paires à optimiser')
    parser.add_argument('--trials', type=int, default=100, help="Nombre d'essais d'optimisation")
    parser.add_argument('--metric', type=str, default='sharpe_ratio', help='Métrique à optimiser (sharpe_ratio, profit_factor, etc)')
    parser.add_argument('--days', type=int, default=90, help="Nombre de jours d'historique à utiliser")
    parser.add_argument('--capital', type=float, default=10000, help='Capital initial pour le backtest')
    parser.add_argument('--testnet', action='store_true', help='Utiliser le testnet Binance')
    parser.add_argument('--save-config', action='store_true', help='Sauvegarder la configuration optimale')
    
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('OptimizeWeights')
    
    # Charger les variables d'environnement
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    print("\n" + "="*60)
    print("🧬 OPTIMISATION DES POIDS DES INDICATEURS")
    print("="*60)
    print(f"📊 Métrique: {args.metric}")
    print(f"🔄 Essais: {args.trials}")
    print(f"📅 Historique: {args.days} jours")
    print(f"💰 Capital: {args.capital} USDT")
    print("="*60 + "\n")
    
    try:
        # Créer l'exchange connector
        logger.info("Connexion à l'exchange...")
        exchange = ExchangeConnector(
            exchange_name='binance',
            testnet=args.testnet
        )
        await exchange.connect(api_key=api_key, api_secret=api_secret)
        
        # Créer l'optimiseur
        optimizer = WeightOptimizer(testnet=args.testnet)
        optimizer.exchange = exchange
        optimizer.optimization_metric = args.metric
        
        # Calculer les dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Optimiser une ou plusieurs paires
        if args.symbols:
            # Optimisation multiple
            logger.info(f"🔄 Optimisation de {len(args.symbols)} paires...")
            print("\nPaires à optimiser:")
            for symbol in args.symbols:
                print(f"  - {symbol}")
            print()
            
            results = optimizer.optimize_multiple_pairs(
                symbols=args.symbols,
                n_trials=args.trials,
                days_back=args.days,
                initial_capital=args.capital
            )
            
            # Afficher le résumé
            print_summary(results)
            
            # Sauvegarder si demandé
            if args.save_config:
                save_optimized_config(results)
                
        else:
            # Optimisation simple
            logger.info(f"🔄 Optimisation de {args.symbol}...")
            
            result = await optimizer.optimize(
                symbol=args.symbol,
                start_date=start_date,
                end_date=end_date,
                n_trials=args.trials,
                initial_capital=args.capital
            )
            
            # Afficher le résumé
            print_summary({args.symbol: result})
            
            # Sauvegarder si demandé
            if args.save_config:
                save_optimized_config({args.symbol: result})
        
        # Sauvegarder tous les résultats
        optimizer.save_results('optimization_history.json')
        
        print("\n✅ Optimisation terminée avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'optimisation: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Fermer la connexion
        if 'exchange' in locals():
            await exchange.close()

def print_summary(results):
    """Affiche un résumé des résultats"""
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES OPTIMISATIONS")
    print("="*60)
    
    for symbol, result in results.items():
        print(f"\n🪙 {symbol}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:>8.3f}")
        print(f"  Return Total: {result.total_return:>8.2f}%")
        print(f"  Win Rate:     {result.win_rate:>8.1f}%")
        print(f"  Max Drawdown: {result.max_drawdown:>8.2f}%")
        print(f"  Profit Factor:{result.profit_factor:>8.2f}")
        
        print("\n  Poids optimaux:")
        for indicator, weight in sorted(result.best_weights.items(), 
                                      key=lambda x: x[1], reverse=True):
            bar = "█" * int(weight * 20)
            print(f"    {indicator:12} {bar:<20} {weight:>6.1%}")

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
    asyncio.run(main())