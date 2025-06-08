#!/usr/bin/env python
"""
Script de migration pour organiser correctement le projet crypto-trading-bot
"""

import os
import shutil
from pathlib import Path

# Mapping des fichiers à créer/déplacer
FILE_MAPPINGS = {
    # Core modules
    'src/core/trading_bot.py': 'trading_bot.py',
    'src/core/weighted_score_engine.py': 'weighted_score_engine.py',
    'src/core/watchlist_scanner.py': 'watchlist_scanner.py',
    'src/core/backtester.py': 'backtester.py',
    'src/core/weight_optimizer.py': 'weight_optimizer.py',
    
    # Exchange modules
    'src/exchanges/exchange_connector.py': 'exchange_connector.py',
    
    # Strategy modules
    'src/strategies/strategy.py': 'strategy.py',
    
    # Web modules
    'src/web/dashboard.py': 'dashboard.py',
    
    # Scripts
    'scripts/optimize_weights.py': 'optimize_weights.py',
    'scripts/run_backtest.py': 'run_backtest.py',
    
    # Tests
    'tests/test_strategies.py': 'test_strategies.py',
    
    # Config files
    'setup.py': 'setup.py',
}

# Fichiers à supprimer (obsolètes)
FILES_TO_DELETE = [
    'src/core/analyzer.py',
    'src/core/trader.py',
    'src/strategies/base.py',
    'src/strategies/ma_cross.py',
    'src/models/candle.py',
    'src/models/order.py',
    'src/models/position.py',
]

def create_init_files():
    """Crée les fichiers __init__.py nécessaires"""
    init_content = {
        'src/__init__.py': '',
        'src/core/__init__.py': '''"""Core modules for crypto trading bot"""

from src.trading_bot import TradingBot
from src.weighted_score_engine import WeightedScoreEngine
from src.multi_pair_manager import MultiPairManager
from src.watchlist_scanner import WatchlistScanner
from src.backtester import Backtester
from src.weight_optimizer import WeightOptimizer
from src.market_data import MarketData
from src.risk_manager import RiskManager

__all__ = [
    'TradingBot',
    'WeightedScoreEngine',
    'MultiPairManager',
    'WatchlistScanner',
    'Backtester',
    'WeightOptimizer',
    'MarketData',
    'RiskManager',
]
''',
        'src/exchanges/__init__.py': '''"""Exchange connectors"""

from src.exchange_connector import ExchangeConnector

__all__ = ['ExchangeConnector']
''',
        'src/strategies/__init__.py': '''"""Trading strategies"""

from src.strategy import Strategy, MultiSignalStrategy
from src.multi_signal import MultiSignalStrategy as MultiSignal

__all__ = ['Strategy', 'MultiSignalStrategy', 'MultiSignal']
''',
        'src/utils/__init__.py': '''"""Utility functions"""

from src.helpers import *
from src.indicators import *
''',
        'src/web/__init__.py': '''"""Web interface modules"""

from src.dashboard import create_app

__all__ = ['create_app']
''',
        'tests/__init__.py': '',
    }
    
    for filepath, content in init_content.items():
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created {filepath}")

def update_imports():
    """Met à jour les imports dans les fichiers existants"""
    # Mise à jour de multi_signal.py
    multi_signal_path = 'src/strategies/multi_signal.py'
    if os.path.exists(multi_signal_path):
        with open(multi_signal_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remplacer les imports
        content = content.replace(
            'from src.core.weighted_score_engine import WeightedScoreEngine',
            'from srccore.weighted_score_engine import WeightedScoreEngine'
        )
        
        with open(multi_signal_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated imports in {multi_signal_path}")
    
    # Mise à jour de multi_pair_manager.py
    manager_path = 'src/core/multi_pair_manager.py'
    if os.path.exists(manager_path):
        with open(manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = content.replace(
            'from src.strategies.multi_signal import MultiSignalStrategy',
            'from srcstrategies.multi_signal import MultiSignalStrategy'
        )
        
        with open(manager_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated imports in {manager_path}")

def create_run_bot_wrapper():
    """Crée un nouveau run_bot.py qui utilise les bons imports"""
    content = '''#!/usr/bin/env python
"""
Point d'entrée principal du bot de trading
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
from dotenv import load_dotenv
from src.core.trading_bot import TradingBot

def main():
    # Parser d'arguments
    parser = argparse.ArgumentParser(description='Bot de trading crypto')
    parser.add_argument('--mode', choices=['paper', 'real'], default='paper',
                       help='Mode de trading (paper ou real)')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='Fichier de configuration')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Niveau de logging')
    
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Créer et démarrer le bot
    bot = TradingBot(
        config_file=args.config,
        paper_trading=(args.mode == 'paper')
    )
    
    try:
        bot.start()
    except KeyboardInterrupt:
        logging.info("Arrêt du bot...")
        bot.stop()
    except Exception as e:
        logging.error(f"Erreur critique: {e}")
        bot.stop()
        raise

if __name__ == "__main__":
    main()
'''
    
    with open('run_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created new run_bot.py")

def create_optimize_script():
    """Crée le script d'optimisation avec les bons imports"""
    content = '''#!/usr/bin/env python
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
        help='Nombre d\'essais (défaut: 100)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Nombre de jours d\'historique (défaut: 30)'
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
    
    print("\\n" + "="*60)
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
        
        print("\\n✅ Optimisation terminée avec succès!")
        
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
    
    print(f"\\n💾 Configuration optimale sauvegardée dans {filename}")

if __name__ == "__main__":
    main()
'''
    
    Path('scripts').mkdir(exist_ok=True)
    with open('scripts/optimize_weights.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Created scripts/optimize_weights.py")

def main():
    """Fonction principale de migration"""
    print("🚀 Début de la migration du projet...")
    
    # 1. Créer les fichiers __init__.py
    create_init_files()
    
    # 2. Mettre à jour les imports
    update_imports()
    
    # 3. Créer les wrappers avec les bons imports
    create_run_bot_wrapper()
    create_optimize_script()
    
    # 4. Supprimer les fichiers obsolètes
    for filepath in FILES_TO_DELETE:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🗑️  Deleted {filepath}")
    
    print("\n✅ Migration terminée!")
    print("\n📝 Prochaines étapes:")
    print("1. Copier les fichiers des artifacts dans les bons répertoires:")
    print("   - trading_bot.py → src/core/")
    print("   - weighted_score_engine.py → src/core/")
    print("   - exchange_connector.py → src/exchanges/")
    print("   - strategy.py → src/strategies/")
    print("   - backtester.py → src/core/")
    print("   - weight_optimizer.py → src/core/")
    print("   - watchlist_scanner.py → src/core/")
    print("   - dashboard.py → src/web/")
    print("   - test_strategies.py → tests/")
    print("\n2. Installer les dépendances manquantes:")
    print("   pip install optuna matplotlib")
    print("\n3. Tester le projet:")
    print("   python run_bot.py --paper")

if __name__ == "__main__":
    main()