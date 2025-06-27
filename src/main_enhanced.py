#!/usr/bin/env python3
# run_enhanced.py

import asyncio
import argparse
import logging
import sys
import signal
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from src.core.trading_bot import TradingBot
from src.core.logger import log_info, log_error

# Variable globale pour le bot
bot_instance = None
dashboard_task = None

async def shutdown(loop):
    """Arrêt propre de l'application"""
    log_info("🛑 Arrêt demandé...")
    
    # Arrêter le bot
    if bot_instance:
        await bot_instance.shutdown()
    
    # Arrêter le dashboard
    if dashboard_task and not dashboard_task.done():
        dashboard_task.cancel()
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass
    
    # Arrêter toutes les tâches restantes
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def handle_exception(loop, context):
    """Gestionnaire d'exceptions"""
    exception = context.get('exception')
    if isinstance(exception, asyncio.CancelledError):
        return
    log_error(f"Exception: {context}")

async def run_dashboard():
    """Lance le dashboard"""
    try:
        import uvicorn
        from src.web.modern_dashboard import ModernDashboard
        dashboard = ModernDashboard(bot_instance)
        config = uvicorn.Config(
            dashboard.app,
            host="127.0.0.1",
            port=8000,
            log_level="warning",
            loop="asyncio"
        )
        server = uvicorn.Server(config)
        await server.serve()
    except asyncio.CancelledError:
        log_info("Dashboard arrêté")

async def main():
    """Fonction principale"""
    global bot_instance, dashboard_task
    
    parser = argparse.ArgumentParser(description='Enhanced Trading Bot')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper')
    parser.add_argument('--no-dashboard', action='store_true', 
                       help='Désactiver le dashboard')
    args = parser.parse_args()
    
    load_dotenv()
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Créer le bot
    bot_instance = TradingBot(paper_trading=(args.mode == 'paper'))
    
    try:
        # Initialiser le bot
        if not await bot_instance.initialize():
            log_error("Échec de l'initialisation")
            return
        
        # Lancer le dashboard
        if not args.no_dashboard:
            dashboard_task = asyncio.create_task(run_dashboard())
            log_info("📊 Dashboard: http://localhost:8000")
        
        # Démarrer le bot
        await bot_instance.start()
        
        # Attendre l'arrêt
        await bot_instance._shutdown_event.wait()
        
    except KeyboardInterrupt:
        log_info("⌨️  Interruption clavier détectée")
    finally:
        await shutdown(asyncio.get_event_loop())
        print("✅ Arrêt propre terminé.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Arrêt du bot")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1)