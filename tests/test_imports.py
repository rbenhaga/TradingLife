#!/usr/bin/env python3
"""
Script de test rapide des imports avec PYTHONPATH corrigé
"""

import sys
import os
from pathlib import Path

# IMPORTANT: Ajouter le répertoire parent au PYTHONPATH
# Ceci permet à Python de trouver le module 'src'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"📁 Répertoire du projet: {project_root}")
print(f"📋 PYTHONPATH: {sys.path[0]}")

def test_imports():
    """Test tous les imports critiques"""
    print("\n🔍 Test des imports...")
    
    imports_to_test = [
        "from src.core.trading_bot import TradingBot",
        "from src.core.weighted_score_engine import WeightedScoreEngine",
        "from src.core.multi_pair_manager import MultiPairManager",
        "from src.core.watchlist_scanner import WatchlistScanner",
        "from src.core.risk_manager import RiskManager",
        "from src.core.market_data import MarketData",
        "from src.exchanges.exchange_connector import ExchangeConnector",
        "from src.strategies.strategy import Strategy",
        "from src.strategies.multi_signal import MultiSignalStrategy",
    ]
    
    success = 0
    failed = []
    
    for import_str in imports_to_test:
        try:
            exec(import_str)
            print(f"✅ {import_str}")
            success += 1
        except ImportError as e:
            print(f"❌ {import_str} - {e}")
            failed.append((import_str, str(e)))
        except Exception as e:
            print(f"⚠️  {import_str} - Erreur: {type(e).__name__}: {e}")
            failed.append((import_str, str(e)))
    
    print(f"\n📊 Résultat: {success}/{len(imports_to_test)} imports réussis")
    
    if failed:
        print("\n❌ Imports échoués:")
        for imp, err in failed:
            print(f"  - {imp}")
            print(f"    Erreur: {err}")
    
    return success == len(imports_to_test)

def check_project_structure():
    """Vérifie la structure du projet"""
    print("\n📂 Vérification de la structure du projet...")
    
    required_files = [
        "src/__init__.py",
        "src/core/__init__.py",
        "src/core/trading_bot.py",
        "src/core/weighted_score_engine.py",
        "src/core/multi_pair_manager.py",
        "src/core/watchlist_scanner.py",
        "src/core/risk_manager.py",
        "src/core/market_data.py",
        "src/strategies/__init__.py",
        "src/strategies/strategy.py",
        "src/strategies/multi_signal.py",
        "src/exchanges/__init__.py",
        "src/exchanges/exchange_connector.py",
    ]
    
    missing = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MANQUANT")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  {len(missing)} fichiers manquants")
        return False
    else:
        print("\n✅ Tous les fichiers requis sont présents")
        return True

def main():
    print("="*60)
    print("🧪 TEST RAPIDE DES IMPORTS")
    print("="*60)
    
    # Vérifier d'abord la structure
    structure_ok = check_project_structure()
    
    if not structure_ok:
        print("\n⚠️  La structure du projet n'est pas complète.")
        print("Certains fichiers doivent être créés ou copiés.")
    
    # Tester les imports
    imports_ok = test_imports()
    
    if imports_ok:
        print("\n🎉 Tous les tests réussis!")
        print("\nProchaine étape: Lancer le test d'intégration complet")
        print("python scripts/test_integration.py")
    else:
        print("\n⚠️  Certains imports ont échoué.")
        print("\nActions recommandées:")
        print("1. Vérifier que tous les fichiers sont présents")
        print("2. Vérifier le contenu des fichiers __init__.py")
        print("3. Corriger les erreurs d'import dans les modules")

if __name__ == "__main__":
    main()
