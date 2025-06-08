#!/usr/bin/env python3
"""
Script pour corriger automatiquement les imports dans le projet TradingLife
"""

import os
import re
from pathlib import Path

# Mapping des corrections d'imports
IMPORT_FIXES = {
    # Imports relatifs à corriger
    r'from \.\.(.*?) import': r'from src\1 import',
    r'from \.(.*?) import': r'from src.\1 import',
    
    # Classes renommées ou déplacées
    r'from src\.core\.watchlist import DynamicWatchlist': 
        'from src.core.watchlist_scanner import WatchlistScanner',
    
    # Standardisation des imports de strategies
    r'from src\.strategies\.base import': 
        'from src.strategies.strategy import',
}

# Fichiers à exclure
EXCLUDE_PATTERNS = [
    '__pycache__',
    '.git',
    'venv',
    'env',
    '.pytest_cache'
]

def should_process_file(filepath: Path) -> bool:
    """Vérifie si le fichier doit être traité"""
    # Exclure les répertoires
    for pattern in EXCLUDE_PATTERNS:
        if pattern in str(filepath):
            return False
    
    # Traiter seulement les fichiers Python
    return filepath.suffix == '.py'

def fix_imports_in_file(filepath: Path) -> int:
    """
    Corrige les imports dans un fichier
    
    Returns:
        Nombre de corrections effectuées
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes_count = 0
        
        # Appliquer chaque correction
        for pattern, replacement in IMPORT_FIXES.items():
            new_content, n = re.subn(pattern, replacement, content)
            if n > 0:
                content = new_content
                fixes_count += n
        
        # Écrire le fichier seulement si modifié
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Corrigé {fixes_count} imports dans: {filepath}")
        
        return fixes_count
        
    except Exception as e:
        print(f"❌ Erreur dans {filepath}: {e}")
        return 0

def add_missing_init_files(root_dir: Path):
    """Ajoute les fichiers __init__.py manquants"""
    init_contents = {
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
        'src/strategies/__init__.py': '''"""Trading strategies"""

from src.strategy import Strategy, MultiSignalStrategy
from src.multi_signal import MultiSignalStrategy as MultiSignal

__all__ = ['Strategy', 'MultiSignalStrategy', 'MultiSignal']
''',
        'src/exchanges/__init__.py': '''"""Exchange connectors"""

from src.exchange_connector import ExchangeConnector

__all__ = ['ExchangeConnector']
''',
        'src/utils/__init__.py': '''"""Utility functions"""

from src.helpers import *
from src.indicators import *
''',
        'src/web/__init__.py': '''"""Web interface modules"""

from src.dashboard import create_app

__all__ = ['create_app']
'''
    }
    
    for filepath, content in init_contents.items():
        full_path = root_dir / filepath
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Créé: {filepath}")

def main():
    """Fonction principale"""
    print("🔧 Correction des imports dans le projet TradingLife")
    print("=" * 50)
    
    # Déterminer le répertoire racine
    root_dir = Path.cwd()
    if not (root_dir / 'src').exists():
        print("❌ Le répertoire 'src' n'existe pas. Êtes-vous dans le bon répertoire?")
        return
    
    # Ajouter les __init__.py manquants
    print("\n📝 Ajout des fichiers __init__.py manquants...")
    add_missing_init_files(root_dir)
    
    # Corriger les imports
    print("\n🔄 Correction des imports...")
    total_fixes = 0
    
    for filepath in root_dir.rglob('*.py'):
        if should_process_file(filepath):
            fixes = fix_imports_in_file(filepath)
            total_fixes += fixes
    
    print(f"\n✨ Terminé! {total_fixes} corrections effectuées.")
    
    # Corrections manuelles nécessaires
    print("\n⚠️  Corrections manuelles nécessaires:")
    print("1. Dans multi_pair_manager.py:")
    print("   - Remplacer 'DynamicWatchlist' par 'WatchlistScanner'")
    print("   - Adapter les noms de méthodes si nécessaire")
    print("\n2. Vérifier que tous les imports fonctionnent avec:")
    print("   python -m py_compile src/**/*.py")

if __name__ == "__main__":
    main() 