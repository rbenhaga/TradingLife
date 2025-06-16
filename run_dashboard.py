#!/usr/bin/env python
"""
Script principal pour lancer le bot avec le dashboard web
"""

import asyncio
import argparse
from src.core.trading_bot import TradingBot
from src.web.dashboard import Dashboard, dashboard_instance
import uvicorn
from src.core.logger import log_info, log_error

async def main():
    """Fonction principale"""
    # Configuration des arguments
    parser = argparse.ArgumentParser(description="Lance le bot de trading avec dashboard")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                      help="Mode de trading (paper/live)")
    parser.add_argument("--config", default="config/config.json",
                      help="Chemin vers le fichier de configuration")
    parser.add_argument("--host", default="localhost",
                      help="Host pour le dashboard")
    parser.add_argument("--port", type=int, default=8000,
                      help="Port pour le dashboard")
    args = parser.parse_args()
    
    try:
        # Initialisation du bot
        bot = TradingBot(
            config_path=args.config,
            paper_trading=(args.mode == "paper")
        )
        
        # Initialiser le dashboard avec le bot
        dashboard_instance.set_bot(bot)
        
        # Démarrer la boucle de mise à jour du dashboard
        asyncio.create_task(dashboard_instance.start_update_loop())
        
        log_info(f"Dashboard disponible sur http://{args.host}:{args.port}")
        log_info("Identifiants: admin / tradingbot123")
        
        # Démarrer le bot
        if await bot.initialize():
            # Lancer le serveur web dans le même processus
            config = uvicorn.Config(
                dashboard_instance.app,
                host=args.host,
                port=args.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
        else:
            log_error("L'initialisation du bot a échoué. Consulte les logs pour plus de détails.")
        
    except Exception as e:
        log_error(f"Erreur lors du démarrage: {e}")
        raise
    finally:
        if 'bot' in locals():
            await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 