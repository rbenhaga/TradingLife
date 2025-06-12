#!/usr/bin/env python3
"""
Script de test complet du TradingBot amélioré
tests/test_trading_bot.py
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import json
from colorama import init, Fore, Style

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.core.trading_bot import TradingBot, BotState
from src.core.logger import log_info, log_error
from src.core.websocket_market_feed import WebSocketMarketFeed, DataType

# Initialiser colorama
init()

def print_success(msg):
    print(f"{Fore.GREEN}✅ {msg}{Style.RESET_ALL}")

def print_error(msg):
    print(f"{Fore.RED}❌ {msg}{Style.RESET_ALL}")

def print_info(msg):
    print(f"{Fore.BLUE}ℹ️  {msg}{Style.RESET_ALL}")

def print_warning(msg):
    print(f"{Fore.YELLOW}⚠️  {msg}{Style.RESET_ALL}")


async def test_websocket_connection():
    """Test la connexion WebSocket"""
    print("\n" + "="*60)
    print("🔌 TEST CONNEXION WEBSOCKET")
    print("="*60)
    
    ws_feed = WebSocketMarketFeed(exchange="binance", testnet=True)
    
    try:
        # Connexion
        connected = await ws_feed.connect()
        if connected:
            print_success("Connexion WebSocket établie")
        else:
            print_error("Échec de connexion WebSocket")
            return False
        
        # Test de souscription
        received_updates = []
        
        async def test_callback(update):
            received_updates.append(update)
            print_info(f"Update reçue: {update.symbol} - {update.data_type.value} - Latence: {update.latency_ms:.1f}ms")
        
        # S'abonner à BTC/USDT
        ws_feed.subscribe("BTC/USDT", [DataType.TICKER, DataType.TRADES], test_callback)
        
        # Attendre quelques updates
        print_info("Attente de données (10 secondes)...")
        await asyncio.sleep(10)
        
        # Vérifier les métriques
        metrics = ws_feed.get_metrics()
        print_info(f"Métriques WebSocket:")
        print_info(f"  - Messages reçus: {metrics['message_count']}")
        print_info(f"  - Latence moyenne: {metrics['avg_latency_ms']:.1f}ms")
        print_info(f"  - Erreurs: {metrics['error_count']}")
        
        # Déconnexion
        await ws_feed.disconnect()
        
        if len(received_updates) > 0:
            print_success(f"Test réussi - {len(received_updates)} updates reçues")
            return True
        else:
            print_warning("Aucune update reçue")
            return False
            
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        return False
    finally:
        # S'assurer que la connexion est fermée
        await ws_feed.disconnect()


async def test_bot_initialization():
    """Test l'initialisation du bot"""
    print("\n" + "="*60)
    print("🤖 TEST INITIALISATION DU BOT")
    print("="*60)
    
    # Créer une config de test
    test_config = {
        "exchange": {
            "name": "binance",
            "testnet": True,
            "api_key": "test_api_key",  # Clé API de test
            "api_secret": "test_api_secret",  # Clé secrète de test
            "skip_connection": True  # Skip la connexion à l'exchange
        },
        "trading": {
            "pairs": ["BTC/USDT"],
            "initial_balance": 10000,
            "position_size": 0.01,
            "max_positions": 2,
            "max_pairs": 5,
            "min_volume_usdt": 1000000,
            "stop_loss": 0.02,
            "take_profit": 0.05
        },
        "strategy": {
            "name": "multi_signal",
            "short_window": 5,
            "long_window": 13,
            "trend_window": 50,
            "timeframe": "15m",
            "min_data_points": 50
        },
        "risk_management": {
            "max_drawdown": 0.15,
            "daily_loss_limit": 0.05,
            "position_sizing": "fixed",
            "max_position_size": 0.02,
            "max_daily_loss": 0.05,
            "max_open_positions": 2,
            "default_stop_loss": 0.02,
            "default_take_profit": 0.05,
            "use_trailing_stop": True,
            "trailing_stop_distance": 0.015
        },
        "scanner_interval": 300,
        "strategy_interval": 60,
        "save_state": True,
        "state_file": "data/test_bot_state.json",
        "logging": {
            "level": "INFO",
            "file": "logs/test_trading.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "max_size": 10485760,  # 10MB
            "backup_count": 5
        }
    }
    
    # Sauvegarder la config temporaire
    config_path = "config/test_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    bot = None
    try:
        # Créer le bot
        bot = TradingBot(config_path=config_path, paper_trading=True)
        print_info("Bot créé")
        
        # Initialiser
        init_success = await bot.initialize()
        
        if init_success:
            print_success("Bot initialisé avec succès")
            
            # Vérifier les composants
            status = bot.get_status()
            print_info("Status du bot:")
            print_info(f"  - État: {status['state']}")
            print_info(f"  - Exchange: {status['config']['exchange']}")
            print_info(f"  - Mode: {'PAPER' if status['paper_trading'] else 'LIVE'}")
            
            # Tester pendant 30 secondes
            print_info("\nTest du bot pendant 30 secondes...")
            
            # Démarrer dans une tâche
            bot_task = asyncio.create_task(bot.start())
            
            # Monitoring
            for i in range(6):  # 6 x 5 secondes = 30 secondes
                await asyncio.sleep(5)
                
                status = bot.get_status()
                print_info(f"\n[{i*5}s] État: {status['state']}")
                print_info(f"  - WebSocket: {'Connecté' if status['websocket'].get('connected') else 'Déconnecté'}")
                print_info(f"  - Updates WS: {status.get('websocket', {}).get('message_count', 0)}")
                print_info(f"  - Positions ouvertes: {status['trading']['open_positions']}")
                
                # Vérifier les erreurs
                if status['errors']:
                    print_warning(f"  - Erreurs: {status['errors'][-1]}")
            
            # Arrêter le bot
            print_info("\nArrêt du bot...")
            await bot.shutdown()
            
            # Attendre la fin
            try:
                await asyncio.wait_for(bot_task, timeout=5.0)
            except asyncio.TimeoutError:
                print_warning("Timeout lors de l'arrêt")
            
            print_success("Test du bot terminé")
            return True
            
        else:
            print_error("Échec de l'initialisation du bot")
            return False
            
    except Exception as e:
        print_error(f"Erreur lors du test: {str(e)}")
        return False
    finally:
        # Nettoyer
        if bot:
            await bot.shutdown()
        if os.path.exists(config_path):
            os.remove(config_path)


async def test_paper_trading():
    """Test le paper trading avec des signaux simulés"""
    print("\n" + "="*60)
    print("📊 TEST PAPER TRADING")
    print("="*60)
    
    print_info("Test de paper trading non implémenté dans cette version")
    return True


async def main():
    """Fonction principale des tests"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("🧪 TESTS DU TRADING BOT AMÉLIORÉ")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Vérifier les prérequis
    if not os.getenv('BINANCE_API_KEY'):
        print_warning("BINANCE_API_KEY non définie - certains tests seront limités")
    
    # Exécuter les tests
    tests = [
        ("WebSocket", test_websocket_connection),
        ("Initialisation Bot", test_bot_initialization),
        # ("Paper Trading", test_paper_trading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            print(f"\n🔄 Exécution: {name}")
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Erreur inattendue dans {name}: {str(e)}")
            results.append((name, False))
    
    # Résumé
    print(f"\n{Fore.CYAN}{'='*60}")
    print("📊 RÉSUMÉ DES TESTS")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    for name, result in results:
        status = f"{Fore.GREEN}✅ PASS{Style.RESET_ALL}" if result else f"{Fore.RED}❌ FAIL{Style.RESET_ALL}"
        print(f"{name:.<40} {status}")
    
    print(f"\n{'='*60}")
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    if success_rate == 100:
        print(f"{Fore.GREEN}🎉 TOUS LES TESTS RÉUSSIS! ({success_count}/{total_count}){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  Tests partiellement réussis: {success_count}/{total_count} ({success_rate:.0f}%){Style.RESET_ALL}")


if __name__ == "__main__":
    asyncio.run(main())