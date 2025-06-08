#!/usr/bin/env python3
"""
Script de lancement des tests depuis la racine du projet
Place ce fichier à la racine du projet (D:\crypto-trading-bot\run_tests.py)
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    """Exécute un script Python"""
    print(f"\n🚀 Exécution de {script_path}")
    print("="*60)
    
    # Utiliser le module Python pour avoir le bon PYTHONPATH
    result = subprocess.run(
        [sys.executable, "-m", script_path],
        capture_output=False
    )
    
    return result.returncode == 0

def main():
    print("🧪 LANCEMENT DES TESTS TRADINGLIFE")
    print("="*60)
    
    # Vérifier qu'on est dans le bon répertoire
    if not Path("src").exists():
        print("❌ Erreur: Le répertoire 'src' n'existe pas.")
        print("Assurez-vous d'exécuter ce script depuis la racine du projet.")
        return
    
    # Menu de sélection
    print("\nQuel test voulez-vous exécuter?")
    print("1. Test rapide des imports")
    print("2. Test d'intégration complet")
    print("3. Test de connexion à l'exchange")
    print("4. Tous les tests")
    
    choice = input("\nVotre choix (1-4): ")
    
    scripts = {
        "1": ["scripts.test_imports"],
        "2": ["scripts.test_integration"],
        "3": ["scripts.test_connection"],
        "4": ["scripts.test_imports", "scripts.test_integration"]
    }
    
    if choice in scripts:
        for script in scripts[choice]:
            if not run_script(script):
                print(f"\n❌ Échec du test {script}")
                break
        else:
            print("\n✅ Tous les tests terminés")
    else:
        print("❌ Choix invalide")

if __name__ == "__main__":
    main()