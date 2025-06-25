#!/usr/bin/env python
"""
Script de lancement amélioré avec toutes les fonctionnalités
"""

import asyncio
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.core.trading_bot import TradingBot
from src.web.modern_dashboard import ModernDashboard
from src.core.logger import log_info, log_error


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper')
    parser.add_argument('--optimize-first', action='store_true', 
                       help='Optimiser avant de commencer')
    args = parser.parse_args()
    
    # Créer le bot
    bot = TradingBot(paper_trading=(args.mode == 'paper'))
    
    # Initialiser
    if not await bot.initialize():
        log_error("Échec initialisation")
        return
    
    # Optimisation initiale si demandée
    if args.optimize_first:
        log_info("🔧 Optimisation initiale...")
        await bot._optimization_loop()  # Une seule itération
    
    # Dashboard moderne
    dashboard = ModernDashboard(bot)
    
    # Démarrer tout
    await asyncio.gather(
        bot.start(),
        dashboard.run()
    )


if __name__ == "__main__":
    asyncio.run(main())