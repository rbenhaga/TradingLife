#!/usr/bin/env python3
"""
Point d'entrée principal du bot de trading
"""

import sys
import os
import argparse
import signal
from datetime import datetime
import time
import asyncio
from typing import Dict, List, Optional

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Charger les variables d'environnement AVANT tout import
from dotenv import load_dotenv
load_dotenv()

from config.settings import load_config, validate_config, default_config
from src.core.logger import log_info, log_error, log_debug, log_warning, log_performance
from src.exchanges.binance import create_binance_testnet_exchange, get_testnet_balance, get_testnet_klines
from src.strategies.ma_cross import MovingAverageCrossStrategy
from src.core.multi_pair_manager import MultiPairManager

# Variable globale pour le bot
bot = None

def signal_handler(signum, frame):
    """Gestionnaire de signal pour arrêter proprement le bot"""
    log_info("\nSignal d'arrêt reçu (Ctrl+C)...")
    if bot:
        bot.stop()
    sys.exit(0)

class TradingBot:
    """Classe principale du bot de trading"""
    
    def __init__(self, config_path: str = None, paper_trading: bool = True, multi_pair: bool = True):
        # Utiliser la config par défaut si pas de fichier spécifié
        try:
            if config_path and os.path.exists(config_path):
                self.config = load_config(config_path)
            else:
                log_warning("Utilisation de la configuration par défaut")
                self.config = default_config
        except Exception as e:
            log_warning(f"Erreur lors du chargement de la config: {e}. Utilisation de la config par défaut.")
            self.config = default_config
            
        self.paper_trading = paper_trading
        self.multi_pair = multi_pair
        self.exchange = None
        self.strategy = None
        self.multi_pair_manager = None
        self.running = False
        self.positions = {}  # Positions ouvertes
        self.trades = []
        self.initial_balance = 0
        self.current_balance = 0
        
        log_info(f"Bot initialisé en mode {'PAPER' if paper_trading else 'LIVE'} trading")

    def initialize(self):
        """Initialise les composants du bot"""
        try:
            # Afficher les clés pour debug (masquées)
            api_key = self.config['exchange']['api_key']
            api_secret = self.config['exchange']['api_secret']
            
            if api_key and api_secret:
                log_debug(f"API Key trouvée: {api_key[:8]}...")
                log_debug(f"API Secret trouvé: {api_secret[:8]}...")
            else:
                log_error("Les clés API ne sont pas configurées dans l'environnement")
                log_info("Assurez-vous que BINANCE_API_KEY et BINANCE_API_SECRET sont définies dans .env")
                return False
            
            # Valider la configuration
            if not validate_config(self.config):
                raise ValueError("Configuration invalide")
            log_info("✅ Configuration validée avec succès")
            
            # Initialiser l'exchange
            log_info("Initialisation de l'exchange...")
            self.exchange = create_binance_testnet_exchange()
            
            # Vérifier la connexion avec une requête simple
            self.exchange.fetch_time()
            log_info("✅ Exchange initialisée avec succès")
            
            # Mode multi-paires ou single pair
            if self.multi_pair:
                log_info("Initialisation du gestionnaire multi-paires...")
                self.multi_pair_manager = MultiPairManager(
                    exchange=self.exchange,
                    config=self.config,
                    paper_trading=self.paper_trading
                )
                
                # Initialisation asynchrone
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    coro = self.multi_pair_manager.initialize()
                    task = loop.create_task(coro)
                    loop.run_until_complete(task)
                    result = task.result()
                else:
                    result = loop.run_until_complete(self.multi_pair_manager.initialize())
                if not result:
                    raise Exception("Échec de l'initialisation du gestionnaire multi-paires")
                log_info("✅ Gestionnaire multi-paires initialisé")
            else:
                # Mode single pair (ancien mode)
                log_info("Initialisation de la stratégie single-pair...")
                strategy_config = self.config['strategy']
                self.strategy = MovingAverageCrossStrategy(
                    short_window=strategy_config['short_window'],
                    long_window=strategy_config['long_window'],
                    trend_window=strategy_config['trend_window'],
                    stop_loss_pct=self.config['trading']['stop_loss'],
                    take_profit_pct=self.config['trading']['take_profit'],
                    symbol=self.config['trading']['pairs'][0]
                )
                log_info("✅ Stratégie initialisée avec succès")
            
            # Récupérer la balance initiale
            balances = get_testnet_balance(self.exchange)
            self.initial_balance = float(balances.get('USDT', {}).get('total', 0))
            self.current_balance = self.initial_balance
            log_info(f"💰 Balance initiale: {self.initial_balance:.2f} USDT")
            
            # Afficher d'autres actifs disponibles
            if len(balances) > 1:
                log_info("Autres actifs disponibles:")
                for asset, balance in list(balances.items())[:5]:
                    if asset != 'USDT' and balance['total'] > 0:
                        log_info(f"  - {asset}: {balance['total']:.4f}")
            
            return True
            
        except Exception as e:
            log_error(f"Erreur lors de l'initialisation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    async def run_multi_pair(self):
        """Boucle principale en mode multi-paires"""
        log_info("🚀 Démarrage du bot multi-paires...")
        log_info(f"📊 Capital: {self.multi_pair_manager.available_capital} USDT")
        log_info(f"🎯 Max positions: {self.config['trading']['max_positions']}")
        
        # Compteurs pour les mises à jour périodiques
        loop_count = 0
        
        try:
            while self.running:
                try:
                    loop_count += 1
                    
                    # Mettre à jour la watchlist toutes les 30 boucles (~30 minutes)
                    if loop_count % 30 == 0:
                        await self.multi_pair_manager.update_watchlist()
                    
                    # Mettre à jour les données de marché
                    await self.multi_pair_manager.update_market_data()
                    
                    # Vérifier les signaux
                    signals = await self.multi_pair_manager.check_signals()
                    
                    if signals:
                        log_info(f"🎯 {len(signals)} signaux détectés")
                        await self.multi_pair_manager.execute_signals(signals)
                    
                    # Afficher les performances toutes les 5 boucles
                    if loop_count % 5 == 0:
                        self.display_multi_pair_performance()
                    
                    # Attendre avant la prochaine itération
                    await asyncio.sleep(60)  # 1 minute
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    log_error(f"Erreur dans la boucle multi-paires: {str(e)}")
                    await asyncio.sleep(60)
                    
        except KeyboardInterrupt:
            log_info("\nSignal d'arrêt reçu (Ctrl+C)...")
        finally:
            self.stop()
    
    def run(self):
        """Boucle principale du bot"""
        if not self.initialize():
            return
        
        self.running = True
        
        if self.multi_pair:
            # Mode multi-paires avec asyncio
            try:
                asyncio.run(self.run_multi_pair())
            except KeyboardInterrupt:
                log_info("\nArrêt demandé par l'utilisateur...")
            except Exception as e:
                log_error(f"Erreur inattendue: {str(e)}")
            finally:
                self.stop()
        else:
            # Mode single pair (ancien mode)
            self.run_single_pair()
    
    def run_single_pair(self):
        """Boucle principale en mode single pair (ancien mode)"""
        log_info("🚀 Démarrage du bot single-pair...")
        log_info(f"📊 Paire: {self.config['trading']['pairs'][0]}")
        log_info(f"⏱️  Timeframe: {self.config['strategy']['timeframe']}")
        log_info(f"📈 Stratégie: MA Cross ({self.config['strategy']['short_window']}/{self.config['strategy']['long_window']}/{self.config['strategy']['trend_window']})")
        
        try:
            while self.running:
                try:
                    # Mettre à jour les données de marché
                    if not self.update_market_data():
                        time.sleep(60)
                        continue
                    
                    # Obtenir le signal de trading
                    signal = self.strategy.get_signal()
                    if signal:
                        log_info(f"🎯 Signal de trading détecté: {signal}")
                        if signal == 'BUY' and not self.positions:
                            self.execute_trade('BUY')
                        elif signal == 'SELL' and self.positions:
                            self.execute_trade('SELL')
                    
                    # Afficher les performances toutes les 5 minutes
                    if int(time.time()) % 300 < 60:  # Toutes les 5 minutes
                        self.display_performance()
                    
                    # Attendre l'intervalle suivant
                    time.sleep(60)  # 1 minute
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    log_error(f"Erreur dans la boucle principale: {str(e)}")
                    time.sleep(60)  # Attendre avant de réessayer
                    
        except KeyboardInterrupt:
            log_info("\nSignal d'arrêt reçu (Ctrl+C)...")
        finally:
            self.stop()
    
    def update_market_data(self):
        """Met à jour les données de marché"""
        try:
            # Récupérer les dernières bougies
            symbol = self.config['trading']['pairs'][0]
            timeframe = self.config['strategy']['timeframe']
            limit = max(self.config['strategy']['min_data_points'], 200)
            
            klines = get_testnet_klines(
                self.exchange,
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            # Mettre à jour la stratégie
            self.strategy.update(klines)
            return True
            
        except Exception as e:
            log_error(f"Erreur lors de la mise à jour des données: {str(e)}")
            return False
    
    def execute_trade(self, side: str):
        """Exécute un trade"""
        try:
            symbol = self.config['trading']['pairs'][0]
            
            if self.paper_trading:
                # Simuler un trade en paper trading
                price = self.strategy.get_current_price()
                if price > 0:
                    if side == 'BUY':
                        # Calculer la taille de la position
                        position_size = self.calculate_position_size(price)
                        if position_size > 0:
                            self.positions[symbol] = {
                                'side': 'LONG',
                                'entry_price': price,
                                'size': position_size,
                                'entry_time': datetime.now()
                            }
                            # Mettre à jour la position dans la stratégie
                            self.strategy.current_position = 'LONG'
                            
                            from src.core.logger import log_trade
                            log_trade('BUY', symbol, position_size, price, 'LONG')
                            
                    else:  # SELL
                        if symbol in self.positions:
                            position = self.positions[symbol]
                            pnl = (price - position['entry_price']) * position['size']
                            self.current_balance += pnl
                            
                            self.trades.append({
                                'entry_price': position['entry_price'],
                                'exit_price': price,
                                'size': position['size'],
                                'pnl': pnl,
                                'entry_time': position['entry_time'],
                                'exit_time': datetime.now()
                            })
                            
                            # Mettre à jour la position dans la stratégie
                            self.strategy.current_position = None
                            
                            from src.core.logger import log_trade
                            log_trade('SELL', symbol, position['size'], price, 'CLOSE', profit=pnl)
                            
                            del self.positions[symbol]
            
            else:
                # Exécution réelle des ordres
                if side == 'BUY':
                    order = self.exchange.create_market_buy_order(
                        symbol,
                        self.calculate_position_size(self.strategy.get_current_price())
                    )
                    log_info(f"ORDRE RÉEL - ACHAT: {order}")
                
                elif side == 'SELL':
                    order = self.exchange.create_market_sell_order(
                        symbol,
                        self.positions[symbol]['size']
                    )
                    log_info(f"ORDRE RÉEL - VENTE: {order}")
            
            return True
            
        except Exception as e:
            log_error(f"Erreur lors de l'exécution du trade: {str(e)}")
            return False
    
    def calculate_position_size(self, price: float) -> float:
        """Calcule la taille de la position"""
        try:
            # Utiliser le pourcentage défini dans la config
            risk_amount = self.current_balance * self.config['trading']['position_size']
            return risk_amount / price
        except Exception as e:
            log_error(f"Erreur lors du calcul de la taille de la position: {str(e)}")
            return 0.0
    
    def display_multi_pair_performance(self):
        """Affiche les performances en mode multi-paires"""
        try:
            summary = self.multi_pair_manager.get_performance_summary()
            
            log_performance({
                'total_trades': summary['total_trades'],
                'win_rate': summary['win_rate'],
                'total_pnl': summary['total_pnl'],
                'positions_open': summary['positions_open'],
                'pairs_trading': summary['pairs_trading']
            })
            
            # Afficher les top performers
            if summary['top_performers']:
                log_info("🏆 Top 3 performers:")
                for perf in summary['top_performers']:
                    log_info(
                        f"  {perf['symbol']}: {perf['trades']} trades, "
                        f"WR: {perf['win_rate']:.1f}%, PnL: {perf['pnl']:+.2f} USDT"
                    )
            
            # Afficher les positions ouvertes
            if self.multi_pair_manager.positions:
                log_info(f"\n📊 Positions ouvertes ({len(self.multi_pair_manager.positions)}):")
                for symbol, pos in self.multi_pair_manager.positions.items():
                    age = (datetime.now() - pos['entry_time']).total_seconds() / 60
                    current_price = self.multi_pair_manager.strategies[symbol].get_current_price()
                    pnl = (current_price - pos['entry_price']) * pos['size']
                    pnl_pct = ((current_price / pos['entry_price']) - 1) * 100
                    
                    emoji = "🟢" if pnl > 0 else "🔴"
                    log_info(
                        f"  {emoji} {symbol}: {pnl:+.2f} USDT ({pnl_pct:+.2f}%) - Age: {age:.0f}min"
                    )
                    
        except Exception as e:
            log_error(f"Erreur affichage performances: {str(e)}")

    def display_performance(self):
        """Affiche les performances actuelles"""
        try:
            from src.core.logger import log_performance
            
            # Calculer les métriques
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            losing_trades = len([t for t in self.trades if t['pnl'] <= 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t['pnl'] for t in self.trades)
            
            # Obtenir les indicateurs actuels
            indicators = self.strategy.get_indicators()
            perf_metrics = self.strategy.get_performance_metrics()
            
            # Logger les performances
            log_performance({
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'current_drawdown': perf_metrics.get('current_drawdown', 0)
            })
            
            # Afficher l'état actuel
            log_info(f"💹 Prix: {indicators['current_price']:.2f} | MA{self.config['strategy']['short_window']}: {indicators['short_ma']:.2f} | MA{self.config['strategy']['long_window']}: {indicators['long_ma']:.2f}")
            
            # Afficher la position actuelle
            if self.positions:
                for symbol, position in self.positions.items():
                    current_price = indicators['current_price']
                    pnl = (current_price - position['entry_price']) * position['size']
                    pnl_pct = ((current_price / position['entry_price']) - 1) * 100
                    
                    emoji = "🟢" if pnl > 0 else "🔴"
                    log_info(f"{emoji} Position {symbol}: PnL {pnl:+.2f} USDT ({pnl_pct:+.2f}%)")
            
        except Exception as e:
            log_error(f"Erreur lors de l'affichage des performances: {str(e)}")
    
    def stop(self):
        """Arrête le bot"""
        self.running = False
        log_info("🛑 Arrêt du bot de trading...")
        
        # Afficher le résumé final
        if self.trades:
            total_pnl = sum(t['pnl'] for t in self.trades)
            log_info(f"📊 Résumé final: {len(self.trades)} trades, PnL total: {total_pnl:+.2f} USDT")

def main():
    """Fonction principale"""
    global bot
    
    # Configurer le gestionnaire de signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parser les arguments
    parser = argparse.ArgumentParser(description='Bot de trading crypto')
    parser.add_argument('--config', type=str, default=None, help='Chemin vers le fichier de configuration')
    parser.add_argument('--paper', action='store_true', default=True, help='Activer le mode paper trading')
    parser.add_argument('--live', action='store_true', help='Activer le mode trading réel')
    parser.add_argument('--single', action='store_true', help='Mode single pair (désactive multi-pair)')
    args = parser.parse_args()
    
    # Déterminer le mode
    paper_trading = not args.live
    multi_pair = not args.single
    
    # Créer le bot
    bot = TradingBot(args.config, paper_trading, multi_pair)
    
    try:
        # Démarrer le bot
        bot.run()
    except KeyboardInterrupt:
        log_info("\nArrêt demandé par l'utilisateur...")
        bot.stop()
    except Exception as e:
        log_error(f"Erreur inattendue: {str(e)}")
        bot.stop()
    finally:
        log_info("Bot arrêté proprement")
        sys.exit(0)

if __name__ == "__main__":
    main()