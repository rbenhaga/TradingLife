#!/usr/bin/env python3
"""
Test avec calcul correct des indicateurs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import pandas_ta as ta

from src.core.backtester import Backtester
from src.strategies.strategy import MultiSignalStrategy
from colorama import init, Fore, Style

init()

def calculate_indicators(df):
    """Calcule TOUS les indicateurs nécessaires"""
    print("📊 Calcul des indicateurs techniques...")
    
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_histogram'] = macd['MACDh_12_26_9']
    
    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['bb_middle'] = bbands['BBM_20_2.0']
    df['bb_lower'] = bbands['BBL_20_2.0']
    
    # Moving Averages
    df['ma_fast'] = ta.sma(df['close'], length=10)
    df['ma_slow'] = ta.sma(df['close'], length=20)
    
    # Volume indicators
    df['volume_sma'] = ta.sma(df['volume'], length=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Momentum
    df['momentum'] = ta.mom(df['close'], length=10)
    
    # ATR pour la volatilité
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    print("✅ Indicateurs calculés")
    print(f"Colonnes disponibles: {', '.join(df.columns)}")
    
    # Vérifier quelques valeurs
    print(f"\nDernières valeurs:")
    print(f"- RSI: {df['rsi'].iloc[-1]:.2f}")
    print(f"- MACD: {df['macd'].iloc[-1]:.2f}")
    print(f"- Prix: ${df['close'].iloc[-1]:.2f}")
    
    return df

def load_and_prepare_data(symbol='BTC/USDT', timeframe='15m', days=7):
    """Charge et prépare les données avec indicateurs"""
    print(f"📥 Chargement des données {symbol}...")
    
    exchange = ccxt.binance()
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ {len(df)} périodes chargées")
    
    # IMPORTANT: Calculer les indicateurs
    df = calculate_indicators(df)
    
    # Supprimer les NaN du début
    df = df.dropna()
    print(f"📊 {len(df)} périodes après nettoyage")
    
    return df

def test_backtest_with_indicators():
    """Test de backtest avec indicateurs calculés"""
    print("\n" + "="*60)
    print("🚀 TEST BACKTEST AVEC INDICATEURS")
    print("="*60)
    
    # Charger et préparer les données
    data = load_and_prepare_data('BTC/USDT', '15m', 7)
    
    # Stratégie agressive
    weights = {
        'rsi': 0.25,
        'bollinger': 0.20,
        'macd': 0.20,
        'volume': 0.15,
        'ma_cross': 0.10,
        'momentum': 0.10
    }
    
    strategy = MultiSignalStrategy('BTC/USDT', score_weights=weights)
    strategy.entry_threshold = 0.25  # Bas
    strategy.exit_threshold = -0.20
    
    # Analyser quelques périodes pour debug
    print("\n🔍 Analyse de quelques signaux:")
    for i in range(len(data)-5, len(data)):
        window = data.iloc[max(0, i-100):i+1]
        
        # Obtenir le signal
        result = strategy.analyze(window)
        
        timestamp = data.index[i].strftime('%Y-%m-%d %H:%M')
        price = data['close'].iloc[i]
        rsi = data['rsi'].iloc[i]
        
        print(f"\n{timestamp} - Prix: ${price:.2f}, RSI: {rsi:.1f}")
        print(f"Signal: {result['action']} (score: {result.get('score', 0):.3f})")
    
    # Lancer le backtest
    print("\n📊 Lancement du backtest...")
    backtester = Backtester(strategy, initial_capital=10000)
    result = backtester.run(data)
    
    print(f"\n✅ Résultats:")
    print(f"- Trades: {result.total_trades}")
    print(f"- Return: {result.total_return_pct:.2f}%")
    print(f"- Sharpe: {result.sharpe_ratio:.2f}")
    print(f"- Win Rate: {result.win_rate:.1f}%")
    
    if result.total_trades > 0:
        print(f"\n📈 Premiers trades:")
        for _, trade in result.trades.head(5).iterrows():
            pnl_color = Fore.GREEN if trade['pnl'] > 0 else Fore.RED
            print(f"  {trade['date'].strftime('%m-%d %H:%M')} - "
                  f"{'BUY' if trade['type'] == 'BUY' else 'SELL'} @ ${trade['price']:.2f} "
                  f"→ P&L: {pnl_color}${trade['pnl']:.2f}{Style.RESET_ALL}")

def test_manual_signals():
    """Test manuel pour voir si les signaux sont générés"""
    print("\n🧪 TEST MANUEL DES SIGNAUX")
    print("="*60)
    
    data = load_and_prepare_data('BTC/USDT', '5m', 3)
    
    # Compter les signaux potentiels
    buy_signals = 0
    sell_signals = 0
    
    for i in range(50, len(data)):
        rsi = data['rsi'].iloc[i]
        close = data['close'].iloc[i]
        bb_lower = data['bb_lower'].iloc[i]
        bb_upper = data['bb_upper'].iloc[i]
        macd = data['macd'].iloc[i]
        macd_signal = data['macd_signal'].iloc[i]
        
        # Conditions d'achat simples
        if (rsi < 40 or close < bb_lower * 1.01 or 
            (macd > macd_signal and macd < 0)):
            buy_signals += 1
            if buy_signals <= 5:  # Afficher les 5 premiers
                print(f"🟢 Signal BUY potentiel: {data.index[i].strftime('%m-%d %H:%M')}")
                print(f"   RSI={rsi:.1f}, Prix=${close:.2f}, BB_lower=${bb_lower:.2f}")
        
        # Conditions de vente
        if (rsi > 60 or close > bb_upper * 0.99 or 
            (macd < macd_signal and macd > 0)):
            sell_signals += 1
    
    print(f"\n📊 Résumé:")
    print(f"- Signaux BUY potentiels: {buy_signals}")
    print(f"- Signaux SELL potentiels: {sell_signals}")

if __name__ == "__main__":
    print(f"{Fore.CYAN}{'='*80}")
    print("🔧 TEST COMPLET AVEC INDICATEURS")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    # Test 1: Backtest avec indicateurs
    test_backtest_with_indicators()
    
    # Test 2: Signaux manuels
    test_manual_signals()
    
    print(f"\n{Fore.YELLOW}💡 Si toujours 0 trades:{Style.RESET_ALL}")
    print("1. Vérifier que pandas_ta est installé: pip install pandas-ta")
    print("2. Réduire encore les seuils (0.1)")
    print("3. Utiliser une période plus volatile") 