#!/usr/bin/env python
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
