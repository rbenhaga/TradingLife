#!/usr/bin/env python3
"""
Script de test d'intégration pour vérifier que tous les modules fonctionnent
"""

import sys
import os
from pathlib import Path

# CORRECTION: Ajouter le répertoire PARENT (racine du projet) au PYTHONPATH
# Path(__file__).parent = scripts/
# Path(__file__).parent.parent = racine du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
from colorama import init, Fore, Style

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

async def test_imports():
    """Test l'importation de tous les modules"""
    print("\n" + "="*60)
    print("📦 TEST DES IMPORTS")
    print("="*60)
    
    modules_to_test = [
        ("Core - TradingBot", "from src.core.trading_bot import TradingBot"),
        ("Core - WeightedScoreEngine", "from src.core.weighted_score_engine import WeightedScoreEngine"),
        ("Core - MultiPairManager", "from src.core.multi_pair_manager import MultiPairManager"),
        ("Core - WatchlistScanner", "from src.core.watchlist_scanner import WatchlistScanner"),
        ("Core - RiskManager", "from src.core.risk_manager import RiskManager"),
        ("Core - MarketData", "from src.core.market_data import MarketData"),
        ("Exchanges - ExchangeConnector", "from src.exchanges.exchange_connector import ExchangeConnector"),
        ("Strategies - Strategy", "from src.strategies.strategy import Strategy"),
        ("Strategies - MultiSignal", "from src.strategies.multi_signal import MultiSignalStrategy"),
    ]
    
    success_count = 0
    for name, import_statement in modules_to_test:
        try:
            exec(import_statement)
            print_success(name)
            success_count += 1
        except Exception as e:
            print_error(f"{name}: {str(e)}")
    
    print(f"\n📊 Résultat: {success_count}/{len(modules_to_test)} imports réussis")
    return success_count == len(modules_to_test)

async def test_exchange_connector():
    """Test le connecteur d'exchange"""
    print("\n" + "="*60)
    print("🔌 TEST DU CONNECTEUR D'EXCHANGE")
    print("="*60)
    
    try:
        from src.exchanges.exchange_connector import ExchangeConnector
        
        # Test d'initialisation
        exchange = ExchangeConnector(exchange_id='binance', testnet=True)
        print_success("ExchangeConnector initialisé")
        
        # Test de connexion (sans clés API)
        # Note: Ceci échouera sans clés API valides
        print_info("Test de connexion sans clés API...")
        connected = await exchange.connect()
        
        if connected:
            print_success("Connexion réussie")
        else:
            print_warning("Connexion échouée (normal sans clés API)")
        
        return True
        
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        return False

async def test_weighted_score_engine():
    """Test le moteur de score pondéré"""
    print("\n" + "="*60)
    print("⚖️ TEST DU MOTEUR DE SCORE")
    print("="*60)
    
    try:
        from src.core.weighted_score_engine import WeightedScoreEngine
        
        # Créer une instance
        engine = WeightedScoreEngine()
        print_success("WeightedScoreEngine créé")
        
        # Créer des données de test
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 50000,
            'volume': np.random.randint(1000000, 5000000, 100),
            'timestamp': pd.date_range(end=datetime.now(), periods=100, freq='15min')
        })
        
        # Ajouter des indicateurs factices
        df['rsi'] = 30 + np.random.rand(100) * 40
        df['macd'] = np.random.randn(100)
        df['macd_signal'] = df['macd'].rolling(9).mean()
        df['bb_upper'] = df['close'] + 1000
        df['bb_lower'] = df['close'] - 1000
        df['ma_fast'] = df['close'].rolling(10).mean()
        df['ma_slow'] = df['close'].rolling(20).mean()
        
        # Analyser
        signals = engine.analyze_indicators(df)
        print_success(f"Signaux analysés: {len(signals)} indicateurs")
        
        # Calculer le score
        result = engine.calculate_score(signals)
        print_success(f"Score calculé: {result['score']:.3f}")
        print_info(f"Action recommandée: {result['action']}")
        print_info(f"Confiance: {result['confidence']:.1%}")
        
        return True
        
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_risk_manager():
    """Test le gestionnaire de risque"""
    print("\n" + "="*60)
    print("🛡️ TEST DU RISK MANAGER")
    print("="*60)
    
    try:
        from src.core.risk_manager import RiskManager
        
        # Configuration
        config = {
            'max_position_size': 0.02,
            'max_daily_loss': 0.05,
            'max_open_positions': 3,
            'default_stop_loss': 0.05,
            'default_take_profit': 0.10
        }
        
        # Créer une instance
        risk_manager = RiskManager(config)
        print_success("RiskManager créé")
        
        # Test can_open_position
        can_open, reason = risk_manager.can_open_position('BTC/USDT', 10000)
        print_success(f"Peut ouvrir position: {can_open} ({reason})")
        
        # Test position sizing
        position_size = risk_manager.calculate_position_size(
            'BTC/USDT', 
            signal_strength=0.8,
            capital=10000,
            current_price=50000
        )
        print_success(f"Taille de position calculée: {position_size:.6f} BTC")
        
        # Test stop loss
        stop_loss = risk_manager.calculate_stop_loss('BTC/USDT', 50000)
        print_success(f"Stop loss calculé: {stop_loss:.2f} USDT")
        
        # Test métriques
        metrics = risk_manager.get_risk_metrics(10000)
        print_info(f"Score de risque: {metrics.risk_score:.1f}/100")
        print_info(f"Drawdown actuel: {metrics.current_drawdown:.1%}")
        
        return True
        
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_market_data():
    """Test le module de données de marché"""
    print("\n" + "="*60)
    print("📊 TEST DU MODULE MARKET DATA")
    print("="*60)
    
    try:
        from src.core.market_data import MarketData
        from src.exchanges.exchange_connector import ExchangeConnector
        
        # Créer un mock exchange connector
        class MockExchange:
            async def get_ohlcv(self, symbol, timeframe, limit):
                # Générer des données factices
                data = []
                for i in range(limit):
                    timestamp = datetime.now().timestamp() * 1000 - (i * 60000)
                    price = 50000 + np.random.randn() * 100
                    data.append([
                        timestamp,
                        price,  # open
                        price + np.random.rand() * 100,  # high
                        price - np.random.rand() * 100,  # low
                        price + np.random.randn() * 50,  # close
                        np.random.randint(1000000, 5000000)  # volume
                    ])
                return data[::-1]  # Inverser pour avoir l'ordre chronologique
            
            async def get_ticker(self, symbol):
                return {
                    'symbol': symbol,
                    'last': 50000 + np.random.randn() * 100,
                    'bid': 49950,
                    'ask': 50050,
                    'percentage': np.random.randn() * 5,
                    'quoteVolume': np.random.randint(10000000, 50000000)
                }
        
        # Créer MarketData avec le mock
        market_data = MarketData(MockExchange())
        print_success("MarketData créé")
        
        # Test d'initialisation
        await market_data.initialize(['BTC/USDT', 'ETH/USDT'])
        print_success("Données initialisées pour 2 symboles")
        
        # Test de récupération
        df = market_data.get_ohlcv('BTC/USDT', '15m')
        print_success(f"OHLCV récupéré: {len(df)} bougies")
        
        # Test des indicateurs
        indicators = market_data.calculate_indicators('BTC/USDT')
        print_success(f"Indicateurs calculés: {len(indicators)} indicateurs")
        
        # Afficher quelques indicateurs
        print_info(f"Prix actuel: {indicators.get('current_price', 0):.2f}")
        print_info(f"RSI: {indicators.get('rsi', 0):.2f}")
        print_info(f"Volume ratio: {indicators.get('volume_ratio', 0):.2f}")
        
        # Test conditions de marché
        conditions = market_data.get_market_conditions('BTC/USDT')
        print_success(f"Conditions analysées: {conditions.get('trend', 'UNKNOWN')}")
        
        return True
        
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_strategy():
    """Test les stratégies"""
    print("\n" + "="*60)
    print("🎯 TEST DES STRATÉGIES")
    print("="*60)
    
    try:
        from src.strategies.multi_signal import MultiSignalStrategy
        
        # Créer une stratégie
        strategy = MultiSignalStrategy('BTC/USDT')
        print_success("MultiSignalStrategy créée")
        
        # Créer des données de test
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 50000,
            'volume': np.random.randint(1000000, 5000000, 100),
            'timestamp': pd.date_range(end=datetime.now(), periods=100, freq='15min')
        })
        
        # Ajouter les indicateurs nécessaires
        df['rsi'] = 25  # RSI oversold pour déclencher un signal
        df['macd'] = 100
        df['macd_signal'] = 50  # MACD > signal
        df['bb_upper'] = df['close'] + 2000
        df['bb_lower'] = df['close'] - 2000
        df['ma_fast'] = df['close'].rolling(10).mean()
        df['ma_slow'] = df['close'].rolling(20).mean()
        
        # Tester l'analyse
        result = strategy.analyze(df)
        print_success(f"Analyse effectuée: {result['action']}")
        print_info(f"Confiance: {result.get('confidence', 0):.1%}")
        print_info(f"Raison: {result.get('reason', 'N/A')}")
        
        # Tester should_enter
        signal = strategy.should_enter(df)
        if signal:
            print_success(f"Signal d'entrée détecté: {signal['action']}")
        else:
            print_warning("Pas de signal d'entrée")
        
        return True
        
    except Exception as e:
        print_error(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Fonction principale"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("🧪 TESTS D'INTÉGRATION TRADINGLIFE")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Vérifier qu'on est dans le bon répertoire
    if not Path('src').exists():
        print_error("Le répertoire 'src' n'existe pas. Êtes-vous dans le bon répertoire?")
        return
    
    # Exécuter tous les tests
    tests = [
        ("Imports", test_imports),
        ("Exchange Connector", test_exchange_connector),
        ("Weighted Score Engine", test_weighted_score_engine),
        ("Risk Manager", test_risk_manager),
        ("Market Data", test_market_data),
        ("Strategies", test_strategy)
    ]
    
    results = []
    for name, test_func in tests:
        try:
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
    success_rate = (success_count / total_count) * 100
    
    if success_rate == 100:
        print(f"{Fore.GREEN}🎉 TOUS LES TESTS RÉUSSIS! ({success_count}/{total_count}){Style.RESET_ALL}")
    elif success_rate >= 80:
        print(f"{Fore.YELLOW}⚠️  Tests partiellement réussis: {success_count}/{total_count} ({success_rate:.0f}%){Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}❌ Tests échoués: {success_count}/{total_count} ({success_rate:.0f}%){Style.RESET_ALL}")
    
    # Prochaines étapes
    print(f"\n{Fore.CYAN}📝 PROCHAINES ÉTAPES:{Style.RESET_ALL}")
    if success_rate < 100:
        print("1. Corriger les imports qui échouent")
        print("2. Implémenter les modules manquants")
        print("3. Vérifier la structure des répertoires")
    else:
        print("1. Configurer les clés API dans .env")
        print("2. Tester avec des données réelles")
        print("3. Lancer le bot en mode paper trading")
        print("4. Optimiser les poids des indicateurs")
    
    print(f"\n💡 Conseil: Exécuter 'python fix_imports.py' pour corriger automatiquement les imports")

if __name__ == "__main__":
    asyncio.run(main())