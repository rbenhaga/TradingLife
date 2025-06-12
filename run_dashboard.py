#!/usr/bin/env python
"""
Lance le bot de trading avec le dashboard web
"""

import asyncio
import uvicorn
from multiprocessing import Process
import sys
from pathlib import Path
import argparse
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from src.core.trading_bot import TradingBot
from src.web.dashboard import Dashboard
from src.core.logger import log_info, log_error


def run_dashboard(dashboard: Dashboard, host: str = "0.0.0.0", port: int = 8000):
    """Lance le serveur web dans un process séparé"""
    uvicorn.run(dashboard.app, host=host, port=port)


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Bot de trading avec dashboard')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Mode de trading')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='Fichier de configuration')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host du dashboard')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port du dashboard')
    
    args = parser.parse_args()
    
    # Charger l'environnement
    load_dotenv()
    
    # Créer le bot
    bot = TradingBot(
        config_path=args.config,
        paper_trading=(args.mode == 'paper')
    )
    
    # Créer le dashboard
    dashboard = Dashboard(trading_bot=bot)
    
    # Lancer le dashboard dans un process séparé
    dashboard_process = Process(
        target=run_dashboard,
        args=(dashboard, args.host, args.port)
    )
    dashboard_process.start()
    
    log_info(f"📊 Dashboard disponible sur http://{args.host}:{args.port}")
    log_info("   Identifiants: admin / tradingbot123")
    
    try:
        # Initialiser et démarrer le bot
        if await bot.initialize():
            # Démarrer la boucle de mise à jour du dashboard
            update_task = asyncio.create_task(dashboard.start_update_loop())
            
            # Démarrer le bot
            await bot.start()
        else:
            log_error("Échec de l'initialisation du bot")
            
    finally:
        # Arrêter le dashboard
        dashboard_process.terminate()
        dashboard_process.join()


if __name__ == "__main__":
    asyncio.run(main()) 