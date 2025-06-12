#!/usr/bin/env python
"""
Point d'entrée principal du bot de trading amélioré
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from src.core.trading_bot import TradingBot
from src.core.logger import log_info, log_error


async def main():
    """Fonction principale"""
    # Parser d'arguments
    parser = argparse.ArgumentParser(description='Bot de trading crypto avancé')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Mode de trading (paper ou live)')
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
    
    # Créer le bot
    bot = TradingBot(
        config_path=args.config,
        paper_trading=(args.mode == 'paper')
    )
    
    try:
        # Initialiser
        log_info("="*60)
        log_info(f"🤖 CRYPTO TRADING BOT - Mode {args.mode.upper()}")
        log_info("="*60)
        
        if not await bot.initialize():
            log_error("Échec de l'initialisation")
            return 1
        
        # Afficher le status initial
        status = bot.get_status()
        log_info(f"Configuration: {status['config']}")
        
        # Démarrer le bot
        await bot.start()
        
    except KeyboardInterrupt:
        log_info("Interruption utilisateur détectée")
    except Exception as e:
        log_error(f"Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Arrêt propre
        await bot.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))