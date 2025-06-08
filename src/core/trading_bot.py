"""
Module principal du bot de trading
"""

import asyncio
import json
from typing import Dict, List, Optional
from datetime import datetime
import ccxt

from .logger import log_info, log_error, log_debug, log_trade
from .multi_pair_manager import MultiPairManager
from .watchlist_scanner import WatchlistScanner
from .weighted_score_engine import WeightedScoreEngine
from .risk_manager import RiskManager
from ..strategies.multi_signal import MultiSignalStrategy
from ..exchanges.exchange_connector import ExchangeConnector
from ..utils.helpers import calculate_position_size
from ..utils.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands

class TradingBot:
    """Bot de trading principal"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialise le bot de trading
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.exchange = None
        self.watchlist = None
        self.risk_manager = None
        self.market_data = None
        self.pair_manager = None
        self.running = False
        
        log_info("Bot de trading initialisé")
    
    def _load_config(self, config_path: str) -> Dict:
        """Charge la configuration depuis le fichier JSON"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            log_error(f"Erreur lors du chargement de la config: {str(e)}")
            return {}
    
    async def initialize(self):
        """Initialise tous les composants du bot"""
        try:
            # Initialiser l'exchange
            self.exchange = ExchangeConnector(
                exchange_id=self.config.get('exchange', 'binance'),
                testnet=self.config.get('testnet', True)
            )
            
            # Connexion à l'exchange
            if not await self.exchange.connect(
                api_key=self.config.get('api_key'),
                api_secret=self.config.get('api_secret')
            ):
                raise Exception("Échec de la connexion à l'exchange")
            
            # Initialiser les composants
            self.watchlist = WatchlistScanner(
                exchange=self.exchange,
                min_volume_usdt=self.config.get('min_volume_usdt', 1_000_000),
                top_n=self.config.get('top_pairs', 10)
            )
            
            self.risk_manager = RiskManager(
                max_position_size=self.config.get('max_position_size', 0.1),
                max_daily_loss=self.config.get('max_daily_loss', 0.05),
                max_open_trades=self.config.get('max_open_trades', 5)
            )
            
            self.market_data = MarketData(
                exchange=self.exchange,
                timeframes=self.config.get('timeframes', ['15m', '1h', '4h'])
            )
            
            self.pair_manager = MultiPairManager(
                exchange=self.exchange,
                risk_manager=self.risk_manager,
                market_data=self.market_data
            )
            
            log_info("Tous les composants ont été initialisés avec succès")
            return True
            
        except Exception as e:
            log_error(f"Erreur lors de l'initialisation: {str(e)}")
            return False
    
    async def start(self):
        """Démarre le bot"""
        if self.running:
            log_warning("Le bot est déjà en cours d'exécution")
            return
        
        self.running = True
        log_info("Démarrage du bot de trading...")
        
        try:
            # Démarrer les tâches asynchrones
            tasks = [
                self._run_watchlist_scanner(),
                self._run_pair_manager(),
                self._run_market_data_collector()
            ]
            
            # Exécuter toutes les tâches en parallèle
            await asyncio.gather(*tasks)
            
        except Exception as e:
            log_error(f"Erreur dans la boucle principale: {str(e)}")
            self.running = False
    
    async def stop(self):
        """Arrête le bot"""
        if not self.running:
            return
        
        self.running = False
        log_info("Arrêt du bot...")
        
        # Fermer proprement les connexions
        if self.exchange:
            await self.exchange.close()
        
        # Attendre que toutes les tâches soient terminées
        await asyncio.sleep(1)
        log_info("Bot arrêté")
    
    async def _run_watchlist_scanner(self):
        """Boucle du scanner de watchlist"""
        while self.running:
            try:
                # Scanner le marché pour les meilleures paires
                pairs = await self.watchlist.scan_market()
                
                # Mettre à jour le gestionnaire de paires
                if pairs:
                    await self.pair_manager.update_watchlist(pairs)
                
                # Attendre avant le prochain scan
                await asyncio.sleep(self.config.get('watchlist_interval', 3600))
                
            except Exception as e:
                log_error(f"Erreur dans le scanner de watchlist: {str(e)}")
                await asyncio.sleep(60)
    
    async def _run_pair_manager(self):
        """Boucle du gestionnaire de paires"""
        while self.running:
            try:
                # Mettre à jour les positions
                await self.pair_manager.update_positions()
                
                # Vérifier les signaux et exécuter les trades
                await self.pair_manager.check_signals()
                
                # Attendre avant la prochaine mise à jour
                await asyncio.sleep(self.config.get('update_interval', 60))
                
            except Exception as e:
                log_error(f"Erreur dans le gestionnaire de paires: {str(e)}")
                await asyncio.sleep(60)
    
    async def _run_market_data_collector(self):
        """Boucle du collecteur de données"""
        while self.running:
            try:
                # Mettre à jour les données de marché
                await self.market_data.update_all()
                
                # Attendre avant la prochaine mise à jour
                await asyncio.sleep(self.config.get('data_interval', 60))
                
            except Exception as e:
                log_error(f"Erreur dans le collecteur de données: {str(e)}")
                await asyncio.sleep(60)
    
    def get_status(self) -> Dict:
        """Retourne l'état actuel du bot"""
        return {
            'running': self.running,
            'exchange': self.exchange.exchange_id if self.exchange else None,
            'watchlist': self.watchlist.watchlist if self.watchlist else [],
            'open_positions': self.pair_manager.get_open_positions() if self.pair_manager else [],
            'daily_pnl': self.risk_manager.get_daily_pnl() if self.risk_manager else 0.0,
            'last_update': datetime.now().isoformat()
        } 