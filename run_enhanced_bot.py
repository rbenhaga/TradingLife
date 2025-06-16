#!/usr/bin/env python
"""
Script de lancement du bot de trading amélioré
"""

import os
import sys
import asyncio
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Importer les modules nécessaires
from core.enhanced_trading_bot import EnhancedTradingBot
from core.logger import log_info, log_error

async def main():
    try:
        # Initialiser le bot
        bot = EnhancedTradingBot()
        
        # Démarrer le bot
        if await bot.initialize():
            log_info("✅ Bot amélioré initialisé avec succès")
            await bot.start()
        else:
            log_error("❌ Échec de l'initialisation du bot")
            sys.exit(1)
            
    except KeyboardInterrupt:
        log_info("Arrêt du bot...")
        await bot.shutdown()
    except Exception as e:
        log_error(f"Erreur critique: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 