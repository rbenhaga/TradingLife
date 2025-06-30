"""
Module principal du bot de trading
Version améliorée avec WebSocket et gestion robuste
"""

import asyncio
import json
import signal
import sys
from typing import Dict, List, Optional, Set, cast
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import os
import traceback
import pandas as pd

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.core.logger import log_info, log_error, log_debug, log_warning, log_trade
from src.core.multi_pair_manager import MultiPairManager
from src.core.watchlist_scanner import WatchlistScanner
from src.core.risk_manager import RiskManager
from src.core.market_data import MarketData
from src.core.websocket_market_feed import WebSocketMarketFeed, DataType, MarketUpdate
from src.exchanges.exchange_connector import ExchangeConnector
from src.strategies.ai_enhanced_strategy import AIEnhancedStrategy
from src.notifications.telegram_notifier import TelegramNotifier, NotificationLevel
from src.core.adaptive_backtester import AdaptiveBacktester
from config.settings import load_config, validate_config
from src.core.fast_market_buffer import FastMarketBuffer
from src.core.optimized_weighted_score_engine import OptimizedWeightedScoreEngine
from src.monitoring.prometheus_metrics import PrometheusExporter, measure_latency, decision_latency


class BotState(Enum):
    """États possibles du bot"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class BotStatus:
    """Status complet du bot"""
    state: BotState
    start_time: datetime
    last_update: datetime
    total_trades: int = 0
    open_positions: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    errors: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class TradingBot:
    """Bot de trading principal avec architecture événementielle"""
    
    def __init__(self, config_path: str = "config/config.json", paper_trading: bool = True):
        """
        Initialise le bot de trading
        
        Args:
            config_path: Chemin vers le fichier de configuration
            paper_trading: Mode paper trading (défaut: True)
        """
        # Configuration
        self.config = self._load_and_validate_config(config_path)
        self.paper_trading = paper_trading
        self.market_buffer = FastMarketBuffer(max_symbols=100)
        self.optimized_score_engine = OptimizedWeightedScoreEngine()
        self.metrics_exporter = None
        
        # État du bot
        self.status = BotStatus(
            state=BotState.INITIALIZING,
            start_time=datetime.now(),
            last_update=datetime.now()
        )
        
        # Composants principaux
        self.exchange: Optional[ExchangeConnector] = None
        self.websocket_feed: Optional[WebSocketMarketFeed] = None
        self.watchlist_scanner: Optional[WatchlistScanner] = None
        self.risk_manager: Optional[RiskManager] = None
        self.market_data: Optional[MarketData] = None
        self.pair_manager: Optional[MultiPairManager] = None
        self.notifier: Optional[TelegramNotifier] = None
        self.backtester: Optional[AdaptiveBacktester] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
        # Contrôle d'exécution
        self._main_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()
        self._force_shutdown = False
        
        # Gestion des signaux système
        self._setup_signal_handlers()
        
        log_info(f"TradingBot initialisé - Mode: {'PAPER' if paper_trading else 'LIVE'}")
    
    def _load_and_validate_config(self, config_path: str) -> Dict:
        """Charge et valide la configuration"""
        try:
            config = load_config(config_path)
            
            # Ajouter les clés d'environnement si nécessaire
            if not config['exchange'].get('api_key'):
                config['exchange']['api_key'] = os.getenv('BINANCE_API_KEY', '')
            if not config['exchange'].get('api_secret'):
                config['exchange']['api_secret'] = os.getenv('BINANCE_API_SECRET', '')
            
            # Valider la configuration
            if not validate_config(config):
                raise ValueError("Configuration invalide")
            
            return config
            
        except Exception as e:
            tb = traceback.format_exc()
            log_error(f"Erreur lors du chargement de la config: {str(e)}\nTraceback:\n{tb}")
            raise
    
    def _setup_signal_handlers(self):
        """Configure les gestionnaires de signaux système"""
        def signal_handler(sig, frame):
            log_info("🛑 Signal d'arrêt reçu, fermeture en cours...")
            self._force_shutdown = True
            self._shutdown_event.set()
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, signal_handler)
    
    async def initialize(self) -> bool:
        """
        Initialise tous les composants du bot
        
        Returns:
            True si l'initialisation réussit
        """
        try:
            self.metrics_exporter = PrometheusExporter(port=8000)
            self.metrics_exporter.start()
            log_info("✅ Prometheus démarré sur port 8000")
        except Exception as e:
            log_error(f"Erreur Prometheus: {e}")
            
        try:
            log_info("Initialisation des composants...")
            
            # 1. Exchange connector
            self.exchange = ExchangeConnector(
                exchange_name=self.config['exchange']['name'],
                testnet=self.config['exchange']['testnet'],
                skip_connection=self.config['exchange'].get('skip_connection', False)
            )
            
            if not await self.exchange.connect(
                api_key=self.config['exchange']['api_key'],
                api_secret=self.config['exchange']['api_secret']
            ):
                raise Exception("Échec de connexion à l'exchange")
            
            # 2. WebSocket feed avec configuration
            ws_config = self.config.get('websocket', {})
            log_info(f"Configuration WebSocket: {ws_config}")
            
            try:
                self.websocket_feed = WebSocketMarketFeed(
                    exchange=self.config['exchange']['name'],
                    testnet=self.config['exchange']['testnet'],
                    max_reconnect_attempts=ws_config.get('max_reconnect_attempts', 5)
                )
                
                # Configurer les paramètres WebSocket
                self.websocket_feed.heartbeat_timeout = ws_config.get('heartbeat_timeout', 30)
                
                if not await self.websocket_feed.connect():
                    raise Exception("Échec de connexion WebSocket")
                
                log_info("WebSocket feed initialisé avec succès")
                
            except Exception as e:
                log_error(f"Erreur lors de l'initialisation du WebSocket feed: {str(e)}")
                raise
            
            # 3. Risk Manager
            self.risk_manager = RiskManager(self.config['risk_management'])
            
            # 4. Market Data Manager
            self.market_data = MarketData(
                exchange_connector=self.exchange,
                config=self.config.get('market_data', {})
            )
            
            # 5. Watchlist Scanner
            self.watchlist_scanner = WatchlistScanner(
                exchange_connector=self.exchange,
                min_volume_usdt=self.config['trading'].get('min_volume_usdt', 1_000_000),
                top_n=self.config['trading'].get('max_pairs', 10)
            )
            
            # 6. Multi-Pair Manager
            self.pair_manager = MultiPairManager(
                exchange=self.exchange,
                config=self.config,
                paper_trading=self.paper_trading
            )
            
            # 7. Initialiser les notifications Telegram
            self.notifier = TelegramNotifier()
            if self.notifier.enabled:
                await self.notifier.send_message(
                    "🚀 Bot démarré en mode " + ("PAPER" if self.paper_trading else "LIVE"),
                    NotificationLevel.SUCCESS
                )
            
            # 8. Initialiser le backtester adaptatif
            self.backtester = AdaptiveBacktester(
                initial_capital=self.config['trading']['initial_balance']
            )
            
            # Initialiser les données de marché
            initial_pairs = self.config['trading']['pairs']
            await self.market_data.initialize(initial_pairs)
            
            # S'abonner aux flux WebSocket
            try:
                await self._setup_websocket_subscriptions(initial_pairs)
            except Exception as e:
                log_error(f"Erreur lors de la configuration des abonnements WebSocket: {str(e)}")
                raise
            
            self.status.state = BotState.STOPPED
            log_info("✅ Tous les composants initialisés avec succès")
            
            return True
            
        except Exception as e:
            tb = traceback.format_exc()
            log_error(f"Erreur lors de l'initialisation: {str(e)}\nTraceback:\n{tb}")
            self.status.state = BotState.ERROR
            self.status.errors.append(str(e))
            return False
    
    async def _setup_websocket_subscriptions(self, symbols: List[str]):
        """Configure les abonnements WebSocket"""
        log_info(f"Configuration des abonnements WebSocket pour {len(symbols)} paires")
        
        if not self.websocket_feed:
            raise Exception("WebSocket feed non initialisé")
            
        for symbol in symbols:
            try:
                log_info(f"Abonnement à {symbol}")
                # S'abonner aux données nécessaires
                self.websocket_feed.subscribe(
                    symbol=symbol,
                    data_types=[DataType.TICKER, DataType.TRADES, DataType.ORDERBOOK],
                    callback=self._handle_market_update
                )
                log_info(f"✅ Abonnement réussi pour {symbol}")
            except Exception as e:
                log_error(f"Erreur lors de l'abonnement à {symbol}: {str(e)}")
                raise
        
        log_info(f"✅ Abonné aux flux WebSocket pour {len(symbols)} paires")
    
    async def _handle_market_update(self, update: MarketUpdate):
        """
        Traite les mises à jour du marché en temps réel
        
        Args:
            update: Mise à jour reçue du WebSocket
        """
        try:
            # Mettre à jour les données en cache
            if update.data_type == DataType.TICKER:
                # Mise à jour rapide du ticker
                if self.market_data and hasattr(self.market_data, 'ticker_cache'):
                    self.market_data.ticker_cache[update.symbol] = update.data
                
            elif update.data_type == DataType.ORDERBOOK:
                # Mise à jour du carnet d'ordres
                # Note: _update_orderbook n'existe pas dans MarketData, on skip pour l'instant
                pass
            
            # Incrémenter les métriques
            self.status.metrics['ws_updates'] = self.status.metrics.get('ws_updates', 0) + 1
            
        except Exception as e:
            log_error(f"Erreur traitement update {update.symbol}: {str(e)}")
    
    async def _get_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Récupère les données historiques pour un symbole"""
        try:
            if not self.exchange:
                return pd.DataFrame()
                
            # Utiliser l'exchange connector pour récupérer les données
            limit = days * 24 * 4  # 4 bougies par heure pour du 15m
            ohlcv = await self.exchange.get_ohlcv(symbol, '15m', limit=limit)
            
            if not ohlcv:
                return pd.DataFrame()
            
            # Convertir en DataFrame avec le bon typage
            df = pd.DataFrame(ohlcv)
            df.columns = pd.Index(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            log_error(f"Erreur récupération données historiques {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def _optimization_loop(self):
        """Boucle d'optimisation automatique"""
        while not self._shutdown_event.is_set():
            try:
                if self.status.state != BotState.RUNNING:
                    await asyncio.sleep(1)
                    continue
                log_info("🔧 Début du cycle d'optimisation")
                
                if not self.pair_manager or not self.backtester:
                    await asyncio.sleep(3600)
                    continue
                
                # Obtenir la liste des symboles
                symbols = list(self.pair_manager.strategies.keys()) if hasattr(self.pair_manager, 'strategies') else []
                
                for symbol in symbols:
                    # Récupérer les données historiques
                    data = await self._get_historical_data(symbol, days=30)
                    
                    if data.empty:
                        continue
                    
                    # Optimiser la stratégie
                    if hasattr(self.backtester, 'optimize_strategy'):
                        result = self.backtester.optimize_strategy(
                            symbol, data, n_trials=50
                        )
                        
                        # Appliquer les nouveaux paramètres
                        if isinstance(result, dict) and result.get('score', 0) > 1.0:
                            strategy = AIEnhancedStrategy(symbol, result.get('params', {}))
                            if hasattr(self.pair_manager, 'strategies'):
                                self.pair_manager.strategies[symbol] = strategy
                            log_info(f"✅ Stratégie optimisée pour {symbol}")
                            
                            if self.notifier:
                                await self.notifier.send_message(
                                    f"Stratégie {symbol} optimisée! Score: {result['score']:.2f}",
                                    NotificationLevel.SUCCESS
                                )
                
                # Attendre avant la prochaine optimisation
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=3600 * 6
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
            except asyncio.CancelledError:
                log_debug("Optimization loop annulée")
                break
            except Exception as e:
                log_error(f"Erreur optimisation: {e}")
                await asyncio.sleep(5)
    
    async def start(self):
        """Démarre le bot de trading"""
        if self.status.state not in [BotState.STOPPED, BotState.ERROR]:
            log_warning(f"Impossible de démarrer, état actuel: {self.status.state}")
            return
        
        log_info("🚀 Démarrage du bot de trading...")
        self.status.state = BotState.RUNNING
        self.status.start_time = datetime.now()
        
        try:
            # Créer la tâche principale
            self._main_task = asyncio.create_task(self._main_loop())
            log_info("Tâche principale créée")
            
            # Attendre que la tâche principale soit terminée
            await self._main_task
            
        except asyncio.CancelledError:
            log_info("Bot arrêté par l'utilisateur")
            self.status.state = BotState.STOPPED
        except Exception as e:
            log_error(f"Erreur critique dans le bot: {str(e)}")
            self.status.state = BotState.ERROR
            self.status.errors.append(str(e))
        finally:
            # Nettoyer les ressources
            await self.shutdown()
    
    async def _main_loop(self):
        """Boucle principale du bot"""
        log_info("Boucle principale démarrée")
        
        # Créer les tâches parallèles
        tasks = [
            self._create_monitored_task(self._market_scanner_task(), "Scanner"),
            self._create_monitored_task(self._strategy_loop(), "Strategy"),
            self._create_monitored_task(self._risk_monitor_loop(), "Risk Monitor"),
            self._create_monitored_task(self._performance_tracker_loop(), "Performance"),
            self._create_monitored_task(self._health_check_loop(), "Health Check"),
            self._create_monitored_task(self._optimization_loop(), "Optimizer")  # Ajout de l'optimizer
        ]
        
        log_info(f"✅ {len(tasks)} tâches principales créées")
        
        try:
            # Boucle principale
            while not self._shutdown_event.is_set():
                # Vérifier l'état des tâches
                for i, task in enumerate(tasks):
                    if task.done():
                        if task.exception():
                            log_error(f"Tâche {i} terminée avec erreur: {task.exception()}")
                            # Recréer la tâche
                            task_name = task.get_name()
                            if "Scanner" in task_name:
                                tasks[i] = self._create_monitored_task(self._market_scanner_task(), "Scanner")
                            elif "Strategy" in task_name:
                                tasks[i] = self._create_monitored_task(self._strategy_loop(), "Strategy")
                            elif "Risk Monitor" in task_name:
                                tasks[i] = self._create_monitored_task(self._risk_monitor_loop(), "Risk Monitor")
                            elif "Performance" in task_name:
                                tasks[i] = self._create_monitored_task(self._performance_tracker_loop(), "Performance")
                            elif "Health Check" in task_name:
                                tasks[i] = self._create_monitored_task(self._health_check_loop(), "Health Check")
                            elif "Optimizer" in task_name:
                                tasks[i] = self._create_monitored_task(self._optimization_loop(), "Optimizer")
                
                # Attendre un court instant
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            log_info("Boucle principale annulée")
            raise
        except Exception as e:
            log_error(f"Erreur critique dans la boucle principale: {str(e)}")
            self.status.state = BotState.ERROR
            raise
        finally:
            log_info("Boucle principale terminée")
            # Annuler toutes les tâches
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Attendre que toutes les tâches soient annulées
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _create_monitored_task(self, coro, name: str) -> asyncio.Task:
        """Crée une tâche surveillée"""
        async def monitored():
            try:
                log_info(f"Tâche {name} démarrée")
                await coro
            except asyncio.CancelledError:
                log_info(f"Tâche {name} annulée")
                raise
            except Exception as e:
                log_error(f"Erreur dans tâche {name}: {str(e)}")
                self.status.errors.append(f"{name}: {str(e)}")
                # Ne pas relancer l'exception pour éviter l'arrêt de la boucle principale
                return
        
        task = asyncio.create_task(monitored(), name=name)
        self._tasks.add(task)
        task.add_done_callback(lambda t: self._tasks.discard(t))
        return task
    
    async def _market_scanner_task(self):
        """Tâche de scan du marché"""
        scan_interval = self.config.get('scanner_interval', 300)
        while not self._shutdown_event.is_set():
            try:
                if self.status.state != BotState.RUNNING:
                    await asyncio.sleep(1)
                    continue
                log_debug("Scan du marché en cours...")
                
                # Mettre à jour la watchlist
                if self.watchlist_scanner:
                    await self.watchlist_scanner.update_watchlist()
                
                # Mettre à jour les données de marché
                if self.market_data and hasattr(self.market_data, 'update_all'):
                    await self.market_data.update_all()
                
                # Attendre avant le prochain scan
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=scan_interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
            except asyncio.CancelledError:
                log_debug("Market scanner loop annulée")
                break
            except Exception as e:
                log_error(f"Erreur dans market scanner: {str(e)}")
                await asyncio.sleep(5)
    
    async def _strategy_loop(self):
        """Boucle d'exécution des stratégies"""
        interval = self.config.get('strategy_interval', 60)
        while not self._shutdown_event.is_set():
            try:
                if self.status.state != BotState.RUNNING:
                    await asyncio.sleep(1)
                    continue
                log_debug("Exécution de la boucle de stratégie...")
                
                # Vérifier que tous les composants sont prêts
                if not all([self.pair_manager, self.market_data, self.websocket_feed]):
                    log_warning("Composants non initialisés, attente...")
                    await asyncio.sleep(5)
                    continue
                
                # Mettre à jour les données
                try:
                    if self.pair_manager is not None:
                        await self.pair_manager.update_market_data()
                        log_debug("Données de marché mises à jour")
                except Exception as e:
                    log_error(f"Erreur mise à jour données: {str(e)}")
                    await asyncio.sleep(10)
                    continue
                
                # Vérifier les signaux
                try:
                    signals = {}
                    if self.pair_manager is not None:
                        signals = await self.pair_manager.check_signals()
                    
                    if signals:
                        log_info(f"📊 {len(signals)} signaux détectés")
                        # Exécuter les signaux
                        if self.pair_manager is not None:
                            await self.pair_manager.execute_signals(signals)
                        
                        # Notifier via Telegram
                        if self.notifier:
                            for symbol, signal_data in signals.items():
                                await self.notifier.notify_trade({
                                    'symbol': symbol,
                                    'side': signal_data['signal'],
                                    'price': signal_data['indicators']['current_price'],
                                    'quantity': 0.001,  # À calculer
                                    'confidence': signal_data.get('confidence', 0),
                                    'reason': signal_data.get('reason', '')
                                })
                except Exception as e:
                    log_error(f"Erreur vérification signaux: {str(e)}")
                    await asyncio.sleep(10)
                    continue
                
                # Mettre à jour les métriques
                if self.pair_manager and hasattr(self.pair_manager, 'positions'):
                    self.status.open_positions = len(self.pair_manager.positions)
                self.status.last_update = datetime.now()
                
                # Sauvegarder l'état
                if self.config.get('save_state', True):
                    await self._save_state()
                    log_debug("État du bot sauvegardé")
                
                # Attendre avant la prochaine itération
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
            except asyncio.CancelledError:
                log_debug("Stratégie loop annulée")
                break
            except Exception as e:
                log_error(f"Erreur dans strategy loop: {str(e)}")
                await asyncio.sleep(5)
    
    async def _risk_monitor_loop(self):
        """Boucle de surveillance des risques"""
        interval = 30
        while not self._shutdown_event.is_set():
            try:
                if self.status.state != BotState.RUNNING:
                    await asyncio.sleep(1)
                    continue
                if not self.risk_manager:
                    await asyncio.sleep(interval)
                    continue
                    
                # Récupérer les métriques de risque
                capital = self.config['trading']['initial_balance']
                risk_metrics = self.risk_manager.get_risk_metrics(capital)
                
                # Vérifier les limites
                if risk_metrics.current_drawdown > 0.15:  # 15% drawdown
                    log_warning(f"⚠️ Drawdown élevé: {risk_metrics.current_drawdown:.1%}")
                    
                    if risk_metrics.current_drawdown > 0.20:  # 20% = limite critique
                        log_error("🚨 DRAWDOWN CRITIQUE - Arrêt du trading")
                        await self._pause_trading()
                        
                        if self.notifier:
                            await self.notifier.send_message(
                                "🚨 ALERTE CRITIQUE: Drawdown > 20% - Trading mis en pause",
                                NotificationLevel.ALERT
                            )
                
                if risk_metrics.daily_pnl < -0.05:  # Perte quotidienne > 5%
                    log_warning(f"⚠️ Perte quotidienne élevée: {risk_metrics.daily_pnl:.1%}")
                
                # Mettre à jour les positions avec trailing stops
                if self.pair_manager and self.websocket_feed and hasattr(self.pair_manager, 'positions'):
                    for symbol, position in self.pair_manager.positions.items():
                        ticker = self.websocket_feed.get_ticker(symbol)
                        if ticker:
                            new_stop = self.risk_manager.update_trailing_stop(
                                position, ticker['last']
                            )
                            if new_stop:
                                position['stop_loss'] = new_stop
                                log_debug(f"Trailing stop mis à jour pour {symbol}: {new_stop:.2f}")
                
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
            except asyncio.CancelledError:
                log_debug("Risk monitor loop annulée")
                break
            except Exception as e:
                log_error(f"Erreur dans risk monitor: {str(e)}")
                await asyncio.sleep(5)
    
    async def _performance_tracker_loop(self):
        """Boucle de suivi des performances"""
        interval = 300
        while not self._shutdown_event.is_set():
            try:
                if self.status.state != BotState.RUNNING:
                    await asyncio.sleep(1)
                    continue
                if not self.pair_manager:
                    await asyncio.sleep(interval)
                    continue
                    
                # Calculer les performances
                perf = {}
                if hasattr(self.pair_manager, 'get_performance_summary'):
                    perf = self.pair_manager.get_performance_summary()
                
                # Mettre à jour le status
                self.status.total_trades = perf.get('total_trades', 0)
                self.status.total_pnl = perf.get('total_pnl', 0.0)
                
                # Calculer le PnL quotidien
                daily_pnl = self._calculate_daily_pnl()
                self.status.daily_pnl = daily_pnl
                
                # Logger les performances
                if perf.get('total_trades', 0) > 0:
                    log_info(
                        f"📈 Performance - Trades: {perf['total_trades']} | "
                        f"Win Rate: {perf.get('win_rate', 0):.1f}% | "
                        f"PnL: {perf.get('total_pnl', 0):+.2f} USDT | "
                        f"Daily: {daily_pnl:+.2f} USDT"
                    )
                
                # Envoyer résumé quotidien si c'est l'heure
                if self.notifier and datetime.now().hour == 18 and datetime.now().minute < 5:
                    await self._send_daily_summary()
                
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
            except asyncio.CancelledError:
                log_debug("Performance tracker loop annulée")
                break
            except Exception as e:
                log_error(f"Erreur dans performance tracker: {str(e)}")
                await asyncio.sleep(5)
    
    async def _send_daily_summary(self):
        """Envoie le résumé quotidien via Telegram"""
        try:
            if not self.pair_manager or not self.risk_manager or not self.notifier:
                return
                
            perf = {}
            if hasattr(self.pair_manager, 'get_performance_summary'):
                perf = self.pair_manager.get_performance_summary()
                
            risk_metrics = self.risk_manager.get_risk_metrics(self.config['trading']['initial_balance'])
            
            summary = {
                'capital': self.config['trading']['initial_balance'] + perf.get('total_pnl', 0),
                'daily_pnl': self.status.daily_pnl,
                'daily_pnl_pct': (self.status.daily_pnl / self.config['trading']['initial_balance']) * 100,
                'total_pnl': perf.get('total_pnl', 0),
                'total_pnl_pct': (perf.get('total_pnl', 0) / self.config['trading']['initial_balance']) * 100,
                'total_trades': perf.get('total_trades', 0),
                'wins': perf.get('total_wins', 0),
                'losses': perf.get('total_losses', 0),
                'win_rate': perf.get('win_rate', 0),
                'max_drawdown': risk_metrics.max_drawdown * 100,
                'sharpe_ratio': risk_metrics.sharpe_ratio
            }
            
            if hasattr(self.notifier, 'notify_daily_summary'):
                await self.notifier.notify_daily_summary(summary)
            
        except Exception as e:
            log_error(f"Erreur envoi résumé quotidien: {str(e)}")
    
    async def _health_check_loop(self):
        """Boucle de vérification de santé"""
        interval = 60
        while not self._shutdown_event.is_set():
            try:
                if self.status.state != BotState.RUNNING:
                    await asyncio.sleep(1)
                    continue
                # Vérifier la connexion WebSocket
                if self.websocket_feed:
                    ws_metrics = self.websocket_feed.get_metrics()
                    if not ws_metrics['connected']:
                        log_error("WebSocket déconnecté, tentative de reconnexion...")
                        await self.websocket_feed.connect()
                    
                    # Vérifier la latence moyenne
                    if ws_metrics['avg_latency_ms'] > 200:
                        log_warning(f"Latence moyenne élevée: {ws_metrics['avg_latency_ms']:.0f}ms")
                
                # Vérifier l'exchange
                if self.exchange and hasattr(self.exchange, 'connected') and not self.exchange.connected:
                    log_error("Exchange déconnecté, tentative de reconnexion...")
                    await self.exchange.connect(
                        self.config['exchange']['api_key'],
                        self.config['exchange']['api_secret']
                    )
                
                # Nettoyer les erreurs anciennes
                if len(self.status.errors) > 100:
                    self.status.errors = self.status.errors[-50:]
                
                # Réinitialiser les compteurs quotidiens si nouveau jour
                await self._check_daily_reset()
                
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                
            except asyncio.CancelledError:
                log_debug("Health check loop annulée")
                break
            except Exception as e:
                log_error(f"Erreur dans health check: {str(e)}")
                await asyncio.sleep(5)
    
    async def _pause_trading(self):
        """Met en pause le trading (garde la surveillance active)"""
        self.status.state = BotState.PAUSED
        log_warning("Trading mis en pause")
        
        # Note: close_all_positions n'existe pas dans MultiPairManager
        # On ferme manuellement toutes les positions
        if self.config.get('close_on_pause', False) and self.pair_manager:
            if hasattr(self.pair_manager, 'positions'):
                for symbol in list(self.pair_manager.positions.keys()):
                    if hasattr(self.pair_manager, 'close_position'):
                        await self.pair_manager.close_position(symbol, {})
    
    def _calculate_daily_pnl(self) -> float:
        """Calcule le PnL du jour"""
        if not self.pair_manager or not hasattr(self.pair_manager, 'performance'):
            return 0.0
            
        perf = self.pair_manager.performance
        daily_pnl = 0.0
        
        for symbol, data in perf.items():
            if data.get('last_trade') and isinstance(data['last_trade'], datetime):
                time_since = datetime.now() - data['last_trade']
                if time_since < timedelta(days=1):
                    # Approximation : prendre une portion du PnL total
                    pnl = data.get('pnl', 0)
                    if pnl is not None:
                        daily_pnl += pnl * 0.1
        
        return daily_pnl
    
    async def _check_daily_reset(self):
        """Vérifie et effectue le reset quotidien si nécessaire"""
        now = datetime.now()
        if hasattr(self, '_last_daily_reset'):
            if now.date() > self._last_daily_reset.date():
                log_info("🔄 Reset quotidien des compteurs")
                if self.risk_manager and hasattr(self.risk_manager, 'reset_daily_counters'):
                    self.risk_manager.reset_daily_counters()
                self._last_daily_reset = now
        else:
            self._last_daily_reset = now
    
    async def _save_state(self):
        """Sauvegarde l'état actuel du bot"""
        try:
            if not self.pair_manager:
                return
                
            positions = {}
            if hasattr(self.pair_manager, 'get_positions'):
                positions = self.pair_manager.get_positions()
                
            state = {
                'timestamp': datetime.now().isoformat(),
                'state': self.status.state.value,
                'total_trades': self.status.total_trades,
                'open_positions': self.status.open_positions,
                'total_pnl': self.status.total_pnl,
                'daily_pnl': self.status.daily_pnl,
                'positions': positions,
                'errors': self.status.errors[-10:]  # Garder les 10 dernières erreurs
            }
            
            state_file = Path('data/bot_state.json')
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            log_debug("État du bot sauvegardé")
            
        except Exception as e:
            log_error(f"Erreur sauvegarde état: {str(e)}")
    
    async def shutdown(self):
        """Arrêt propre du bot"""
        log_info("🔴 Début de l'arrêt du bot...")
        self._shutdown_event.set()
        self.status.state = BotState.STOPPING
        if self.websocket_feed:
            log_info("Fermeture des connexions WebSocket...")
            await self.websocket_feed.disconnect()
        log_info(f"Annulation de {len(self._tasks)} tâches...")
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._save_state()
        if self.notifier and self.notifier.enabled:
            await self.notifier.send_message(
                "🔴 Bot arrêté",
                NotificationLevel.INFO
            )
        self.status.state = BotState.STOPPED
        log_info("✅ Bot arrêté proprement")
    
    def get_status(self) -> Dict:
        """Retourne l'état actuel du bot"""
        uptime = datetime.now() - self.status.start_time
        
        # Métriques WebSocket
        ws_metrics = {}
        if self.websocket_feed:
            ws_metrics = self.websocket_feed.get_metrics()
        
        # Performance
        perf = {}
        if self.pair_manager and hasattr(self.pair_manager, 'get_performance_summary'):
            perf = self.pair_manager.get_performance_summary()
        
        pairs_count = 0
        if self.pair_manager and hasattr(self.pair_manager, 'strategies'):
            pairs_count = len(self.pair_manager.strategies)
        
        return {
            'state': self.status.state.value,
            'uptime': str(uptime),
            'start_time': self.status.start_time.isoformat(),
            'last_update': self.status.last_update.isoformat(),
            'paper_trading': self.paper_trading,
            'trading': {
                'total_trades': self.status.total_trades,
                'open_positions': self.status.open_positions,
                'total_pnl': self.status.total_pnl,
                'daily_pnl': self.status.daily_pnl,
                'performance': perf
            },
            'websocket': ws_metrics,
            'errors': self.status.errors[-10:] if self.status.errors else [],
            'config': {
                'exchange': self.config['exchange']['name'],
                'pairs': pairs_count,
                'strategy': self.config['strategy']['name']
            }
        }
    
    # Méthodes pour le dashboard moderne
    def get_portfolio_value(self) -> float:
        """Retourne la valeur totale du portfolio"""
        if not self.pair_manager:
            return self.config['trading']['initial_balance']
        
        perf = {}
        if hasattr(self.pair_manager, 'get_performance_summary'):
            perf = self.pair_manager.get_performance_summary()
        return self.config['trading']['initial_balance'] + perf.get('total_pnl', 0)
    
    def get_open_positions(self) -> List[Dict]:
        """Retourne les positions ouvertes"""
        if not self.pair_manager or not hasattr(self.pair_manager, 'positions'):
            return []
        
        positions = []
        for symbol, pos in self.pair_manager.positions.items():
            ticker = None
            if self.websocket_feed:
                ticker = self.websocket_feed.get_ticker(symbol)
            current_price = ticker['last'] if ticker else pos['entry_price']
            
            pnl = (current_price - pos['entry_price']) * pos['size']
            pnl_pct = ((current_price / pos['entry_price']) - 1) * 100
            
            positions.append({
                'symbol': symbol,
                'side': pos['side'],
                'entry_price': pos['entry_price'],
                'current_price': current_price,
                'size': pos['size'],
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        return positions
    
    def get_daily_pnl(self) -> float:
        """Retourne le P&L du jour"""
        return self.status.daily_pnl
    
    def get_total_pnl(self) -> float:
        """Retourne le P&L total"""
        return self.status.total_pnl
    
    def get_sharpe_ratio(self) -> float:
        """Retourne le Sharpe ratio"""
        if self.risk_manager:
            metrics = self.risk_manager.get_risk_metrics(self.config['trading']['initial_balance'])
            return metrics.sharpe_ratio
        return 0.0
    
    def get_win_rate(self) -> float:
        """Retourne le win rate"""
        if self.pair_manager and hasattr(self.pair_manager, 'get_performance_summary'):
            perf = self.pair_manager.get_performance_summary()
            return perf.get('win_rate', 0)
        return 0.0
    
    def get_profit_factor(self) -> float:
        """Retourne le profit factor"""
        if self.risk_manager and hasattr(self.risk_manager, 'performance_history') and len(self.risk_manager.performance_history) > 0:
            wins = [t['pnl'] for t in self.risk_manager.performance_history if t.get('win', False)]
            losses = [abs(t['pnl']) for t in self.risk_manager.performance_history if not t.get('win', False)]
            
            total_wins = sum(wins) if wins else 0
            total_losses = sum(losses) if losses else 1
            
            return total_wins / total_losses
        return 0.0
    
    def get_max_drawdown(self) -> float:
        """Retourne le max drawdown"""
        if self.risk_manager:
            metrics = self.risk_manager.get_risk_metrics(self.config['trading']['initial_balance'])
            return metrics.max_drawdown * 100
        return 0.0
    
    def get_market_analysis(self) -> Dict:
        """Retourne l'analyse de marché pour toutes les paires"""
        analysis = {}
        
        if self.market_data and self.pair_manager and hasattr(self.pair_manager, 'strategies'):
            for symbol in self.pair_manager.strategies.keys():
                conditions = self.market_data.get_market_conditions(symbol)
                
                # Identifier le régime si on utilise AIEnhancedStrategy
                regime = "UNKNOWN"
                strategy = self.pair_manager.strategies.get(symbol)
                if isinstance(strategy, AIEnhancedStrategy):
                    if hasattr(strategy, 'current_regime') and strategy.current_regime:
                        regime = strategy.current_regime.type
                
                analysis[symbol] = {
                    'regime': regime,
                    'trend': conditions.get('trend', 'UNKNOWN'),
                    'volatility': conditions.get('volatility', 0),
                    'volume': conditions.get('volume_status', 'UNKNOWN'),
                    'score': 0  # À implémenter
                }
        
        return analysis
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Retourne les trades récents"""
        trades = []
        
        if self.risk_manager and hasattr(self.risk_manager, 'performance_history'):
            history = self.risk_manager.performance_history[-limit:]
            
            for trade in reversed(history):
                trades.append({
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'timestamp': trade.get('exit_time', datetime.now()).isoformat(),
                    'pnl': trade['pnl'],
                    'pnl_pct': trade['pnl_pct']
                })
        
        return trades
    
    def get_active_signals(self) -> List[Dict]:
        """Retourne les signaux actifs"""
        signals = []
        
        # Pour l'instant, on retourne une liste vide
        # À implémenter quand les stratégies génèrent des signaux persistants
        
        return signals