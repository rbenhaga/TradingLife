"""
Script pour exécuter les backtests
"""

import asyncio
from src.core.backtester import Backtester
from src.core.logger import log_info, log_error

async def main():
    """Fonction principale"""
    try:
        backtester = Backtester()
        # TODO: Implémenter le backtest
        log_info("Backtest terminé")
    except Exception as e:
        log_error(f"Erreur lors du backtest: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 