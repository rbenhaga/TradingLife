# src/main_enhanced.py
"""
Point d'entrée principal avec toutes les améliorations
"""

import asyncio
import argparse
from pathlib import Path
import sys

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent))

from core.enhanced_trading_bot import EnhancedTradingBot
from core.adaptive_backtester import AdaptiveBacktester
from strategies.ai_enhanced_strategy import AIEnhancedStrategy
from web.modern_dashboard import ModernDashboard
from notifications.telegram_notifier import TelegramNotifier
from core.logger import log_info, log_error


async def main():
    parser = argparse.ArgumentParser(description="Trading Bot Avancé avec IA")
    parser.add_argument('--mode', choices=['backtest', 'optimize', 'live', 'paper'], 
                       default='paper', help='Mode de fonctionnement')
    parser.add_argument('--config', default='config/config.json', 
                       help='Fichier de configuration')
    parser.add_argument('--pairs', nargs='+', default=['BTC/USDT', 'ETH/USDT'], 
                       help='Paires à trader')
    
    args = parser.parse_args()
    
    if args.mode == 'optimize':
        # Mode optimisation
        log_info("🔧 Mode Optimisation Adaptive")
        
        backtester = AdaptiveBacktester()
        
        for pair in args.pairs:
            log_info(f"Optimisation de {pair}...")
            
            # Charger les données historiques
            # data = await load_historical_data(pair, days=180)
            
            # Optimisation adaptive
            # results = await backtester.adaptive_optimization(pair, data)
            
            # Sauvegarder les résultats
            # backtester.save_optimization_results(f"optimization_{pair}.json")
    
    elif args.mode in ['live', 'paper']:
        # Mode trading
        log_info(f"🚀 Démarrage en mode {args.mode.upper()}")
        
        # Créer le bot
        bot = EnhancedTradingBot(
            config_path=args.config,
            paper_trading=(args.mode == 'paper')
        )
        
        # Initialiser
        if not await bot.initialize():
            log_error("Échec de l'initialisation")
            return
        
        # Configurer les notifications Signal
        notifier = TelegramNotifier(
            bot_token="", # A AJOUTER !
            chat_ids=[]
        )
        
        if await notifier.initialize():
            bot.notifier = notifier
            log_info("✅ Notifications Signal configurées")
        
        # Démarrer le dashboard
        dashboard = ModernDashboard(bot)

        # Optimisation initiale si demandée
        if args.optimize_first:
            log_info("🔧 Optimisation initiale...")
            await bot._optimization_loop()  # Une seule itération

        
        # Lancer en parallèle
        await asyncio.gather(
            bot.start(),
            dashboard.run()
        )

if __name__ == "__main__":
    asyncio.run(main())