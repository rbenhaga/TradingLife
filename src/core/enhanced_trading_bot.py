# src/core/enhanced_trading_bot.py
"""
Version améliorée du TradingBot avec les nouveaux composants haute performance
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.core.trading_bot import TradingBot, BotState
from src.core.fast_market_buffer import FastMarketBuffer
from src.strategies.scalping_strategy import ScalpingStrategy
from src.monitoring.prometheus_metrics import PrometheusExporter, measure_latency, decision_latency
from src.core.logger import log_info, log_error


class EnhancedTradingBot(TradingBot):
    """Version améliorée avec composants haute performance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Nouveaux composants
        self.fast_buffer = FastMarketBuffer(max_symbols=200, buffer_size=10000)
        self.prometheus = PrometheusExporter(port=8001)
        self.scalping_strategies = {}
        
    async def initialize(self) -> bool:
        """Initialise avec les nouveaux composants"""
        # Initialisation de base
        if not await super().initialize():
            return False
        
        # Démarrer Prometheus
        self.prometheus.start()
        log_info("✅ Prometheus exporter démarré sur port 8001")
        
        # Initialiser les stratégies de scalping pour les top paires
        for symbol in self.config['trading']['pairs'][:5]:  # Top 5 pour commencer
            self.scalping_strategies[symbol] = ScalpingStrategy(symbol)
            log_info(f"✅ Stratégie scalping initialisée pour {symbol}")
        
        return True
    
    @measure_latency(decision_latency, {'strategy': 'scalping'})
    async def _process_scalping_signal(self, symbol: str):
        """Traite les signaux de scalping avec mesure de latence"""
        
        # Obtenir les données du buffer rapide
        snapshot = self.fast_buffer.get_latest(symbol)
        if not snapshot:
            return
        
        stats = self.fast_buffer.get_stats(symbol, window_ms=1000)
        micro = self.fast_buffer.get_microstructure_signals(symbol)
        
        # Générer le signal
        strategy = self.scalping_strategies.get(symbol)
        if not strategy:
            return
        
        signal = strategy.generate_signal(snapshot, stats, micro)
        
        if signal:
            # Enregistrer dans Prometheus
            self.prometheus.record_signal(
                strategy='scalping',
                signal_type=signal.signal_type.value,
                symbol=symbol,
                confidence=signal.confidence
            )
            
            # Exécuter si confiance suffisante
            if signal.confidence > 0.7:
                await self._execute_scalping_trade(symbol, signal)
    
    async def _execute_scalping_trade(self, symbol, signal):
        """Exécute un trade de scalping (méthode à implémenter selon la logique métier)."""
        log_info(f"Exécution du trade scalping pour {symbol} avec signal {signal.signal_type.value} (confiance: {signal.confidence})")
        # TODO: Ajouter la logique d'exécution réelle ici
    
    async def _enhanced_market_update_handler(self, update):
        """Handler amélioré avec le buffer rapide"""
        # Mise à jour du buffer rapide
        self.fast_buffer.update(
            update.symbol,
            bid=update.data.get('bid', 0),
            ask=update.data.get('ask', 0),
            last=update.data.get('last', 0),
            volume=update.data.get('volume', 0)
        )
        
        # Traiter les signaux de scalping en parallèle
        asyncio.create_task(self._process_scalping_signal(update.symbol))
        
        # Appeler le handler parent
        await super()._handle_market_update(update)

if __name__ == "__main__":
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

    asyncio.run(main())