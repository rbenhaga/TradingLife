#!/usr/bin/env python3
"""
Script de debug pour comprendre pourquoi aucun trade n'est généré
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt

from src.core.backtester import Backtester
from src.strategies.strategy import MultiSignalStrategy
from src.core.weighted_score_engine import WeightedScoreEngine
from colorama import init, Fore, Style

init()

def load_test_data(symbol='BTC/USDT', timeframe='15m', days=7):
    """Charge des données de test"""
    print(f"📊 Chargement des données {symbol}...")
    
    exchange = ccxt.binance()
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ {len(df)} périodes chargées")
    return df

def analyze_signals_debug(strategy, data):
    """Analyse détaillée des signaux"""
    print("\n🔍 ANALYSE DÉTAILLÉE DES SIGNAUX")
    print("="*80)
    
    # Prendre les 10 dernières périodes
    for i in range(max(100, len(data)-10), len(data)):
        window = data.iloc[max(0, i-100):i+1]
        
        if len(window) < 50:
            continue
        
        # Calculer les indicateurs manuellement
        close = window['close']
        high = window['high']
        low = window['low']
        volume = window['volume']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Bollinger Bands
        sma = close.rolling(window=20).mean().iloc[-1]
        std = close.rolling(window=20).std().iloc[-1]
        bb_upper = sma + (2 * std)
        bb_lower = sma - (2 * std)
        current_price = close.iloc[-1]
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = (ema12 - ema26).iloc[-1]
        signal = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]
        
        # Analyser avec la stratégie
        signals = strategy.score_engine.analyze_indicators(window)
        score = strategy.score_engine.calculate_score(signals, symbol='BTC/USDT')
        
        print(f"\n📅 {window.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"Prix: ${current_price:.2f}")
        print(f"RSI: {rsi:.1f} {'(oversold)' if rsi < 30 else '(overbought)' if rsi > 70 else ''}")
        print(f"BB: Lower=${bb_lower:.2f}, Upper=${bb_upper:.2f}")
        print(f"MACD: {macd:.2f}, Signal: {signal:.2f} {'(bullish)' if macd > signal else '(bearish)'}")
        print(f"Score Total: {score.total_score:.3f} (seuil: {strategy.entry_threshold})")
        
        if abs(score.total_score) > 0.1:
            print(f"→ Signal: {Fore.GREEN if score.total_score > 0 else Fore.RED}"
                  f"{score.get_action()}{Style.RESET_ALL}")
            
            # Détail des composants
            for sig in score.signals[:3]:  # Top 3 signaux
                print(f"  - {sig.name}: {sig.weighted_value:.3f} ({sig.reason})")

def test_aggressive_strategy():
    """Test avec une stratégie très agressive"""
    print("\n🚀 TEST STRATÉGIE AGRESSIVE")
    print("="*80)
    
    # Charger les données
    data = load_test_data('BTC/USDT', '15m', 7)
    
    # Créer une stratégie TRÈS agressive
    weights = {
        'rsi': 0.30,
        'bollinger': 0.25,
        'macd': 0.20,
        'volume': 0.10,
        'ma_cross': 0.10,
        'momentum': 0.05
    }
    
    strategy = MultiSignalStrategy('BTC/USDT', score_weights=weights)
    strategy.entry_threshold = 0.15  # TRÈS bas
    strategy.exit_threshold = -0.10  # Sortie rapide
    strategy.rsi_oversold = 40       # Plus sensible
    strategy.rsi_overbought = 60     # Plus sensible
    
    print(f"Configuration agressive:")
    print(f"- Seuil entrée: {strategy.entry_threshold}")
    print(f"- Seuil sortie: {strategy.exit_threshold}")
    print(f"- RSI oversold: {strategy.rsi_oversold}")
    
    # Analyser les signaux
    analyze_signals_debug(strategy, data)
    
    # Backtester
    print("\n📊 BACKTEST")
    print("-"*40)
    
    backtester = Backtester(strategy, initial_capital=10000)
    result = backtester.run(data)
    
    print(f"Trades exécutés: {result.total_trades}")
    print(f"Return: {result.total_return_pct:.2f}%")
    print(f"Win Rate: {result.win_rate:.1f}%")
    
    if result.total_trades > 0:
        print(f"\n📈 Trades:")
        for _, trade in result.trades.head(5).iterrows():
            print(f"  {trade['date'].strftime('%m-%d %H:%M')} - "
                  f"{'BUY' if trade['type'] == 'BUY' else 'SELL'} @ ${trade['price']:.2f}")

def test_simple_strategy():
    """Test avec une stratégie ultra simple (RSI uniquement)"""
    print("\n🎯 TEST STRATÉGIE SIMPLE (RSI)")
    print("="*80)
    
    data = load_test_data('BTC/USDT', '5m', 3)
    
    # Stratégie basée uniquement sur le RSI
    class SimpleRSIStrategy:
        def __init__(self):
            self.symbol = 'BTC/USDT'
            self.positions = []
            
        def analyze(self, data):
            """Analyse simple basée sur RSI"""
            if len(data) < 20:
                return None
                
            # Calculer RSI
            close = data['close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Signal simple
            if rsi < 30:
                return {'action': 'BUY', 'confidence': 0.8, 'reason': f'RSI oversold ({rsi:.1f})'}
            elif rsi > 70:
                return {'action': 'SELL', 'confidence': 0.8, 'reason': f'RSI overbought ({rsi:.1f})'}
            
            return {'action': 'NEUTRAL', 'confidence': 0}
    
    # Simuler manuellement
    strategy = SimpleRSIStrategy()
    trades = 0
    
    for i in range(20, len(data)):
        window = data.iloc[max(0, i-50):i+1]
        signal = strategy.analyze(window)
        
        if signal and signal['action'] != 'NEUTRAL':
            trades += 1
            print(f"{data.index[i].strftime('%m-%d %H:%M')} - "
                  f"{signal['action']} - {signal['reason']}")
    
    print(f"\nTotal trades possibles: {trades}")

def main():
    print(f"{Fore.CYAN}{'='*80}")
    print("🔍 DEBUG BACKTEST - ANALYSE DES SIGNAUX")
    print(f"{'='*80}{Style.RESET_ALL}\n")
    
    # Test 1: Stratégie agressive
    test_aggressive_strategy()
    
    # Test 2: Stratégie simple
    test_simple_strategy()
    
    print(f"\n{Fore.YELLOW}💡 RECOMMANDATIONS:{Style.RESET_ALL}")
    print("1. Utiliser des seuils plus bas (0.1-0.3)")
    print("2. Ajuster les paramètres RSI (35/65 au lieu de 30/70)")
    print("3. Utiliser des timeframes courts (5m, 15m)")
    print("4. Augmenter les poids des indicateurs les plus réactifs")

if __name__ == "__main__":
    main() 