#!/usr/bin/env python
"""
Script de lancement amélioré avec toutes les fonctionnalités
"""

import asyncio
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.trading_bot import TradingBot
from src.web.modern_dashboard import ModernDashboard
from src.core.logger import log_info, log_error

npNaN = np.nan

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
    # dashboard = ModernDashboard(bot)
    
    # Démarrer tout
    await asyncio.gather(
        bot.start(),
        run_dashboard(bot)
    )

async def run_dashboard(bot):
    """Lance le dashboard dans un thread séparé"""
    import uvicorn
    from src.web.modern_dashboard import ModernDashboard

    dashboard = ModernDashboard(bot)
    config = uvicorn.Config(
        dashboard.app,
        host="127.0.0.1",  # Changé de 0.0.0.0
        port=8000,
        log_level="warning"  # Réduire le spam de logs
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Arrêt du bot")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1)