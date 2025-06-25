#!/usr/bin/env python
"""Vérifie les types et corrige les erreurs communes"""
import subprocess
import sys

def run_mypy():
    """Lance mypy pour vérifier les types"""
    print("🔍 Vérification des types avec mypy...")
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "src/", "--ignore-missing-imports"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Aucune erreur de type détectée!")
    else:
        print("❌ Erreurs de type détectées:")
        print(result.stdout)
        
def check_imports():
    """Vérifie que tous les imports sont disponibles"""
    print("\n📦 Vérification des imports...")
    
    missing = []
    for module in ['pandas-ta', 'optuna', 'ccxt']:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - MANQUANT")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Modules manquants: {', '.join(missing)}")
        print("Installer avec: pip install " + " ".join(missing))

if __name__ == "__main__":
    check_imports()
    run_mypy() 