#!/usr/bin/env python3
"""
Script de lancement simplifié du bot de trading
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

# Importer et lancer le bot
from src.main import main

if __name__ == "__main__":
    # Créer les répertoires nécessaires
    (root_dir / "data" / "logs").mkdir(parents=True, exist_ok=True)
    (root_dir / "data" / "cache").mkdir(parents=True, exist_ok=True)
    
    # Lancer le bot
    main()