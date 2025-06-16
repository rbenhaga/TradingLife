"""
Gestionnaire de trading multi-paires
Coordonne les stratégies sur plusieurs paires simultanément
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.core.watchlist_scanner import WatchlistScanner
from src.core.market_data import MarketData
from src.core.logger import log_info, log_debug, log_warning, log_trade, log_error
from .weighted_score_engine import WeightedScoreEngine
from .risk_manager import RiskManager
from ..strategies.strategy import MultiSignalStrategy
from ..utils.helpers import calculate_position_size
from ..utils.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands

class MultiPairManager:
    """Gère le trading sur plusieurs paires simultanément"""
    
    def __init__(self, exchange, config: dict, paper_trading: bool = True):
        """
        Initialise le gestionnaire multi-paires
        
        Args:
            exchange: Instance CCXT
            config: Configuration du bot
            paper_trading: Mode paper trading
        """
        self.exchange = exchange
        self.config = config
        self.paper_trading = paper_trading
        
        # Scanner de volatilité
        self.watchlist_scanner = WatchlistScanner(
            exchange_connector=exchange,
            min_volume_usdt=5_000_000,  # 5M minimum pour commencer
            top_n=10
        )
        
        # Stratégies par paire
        self.strategies = {}
        
        # Positions ouvertes
        self.positions = {}
        
        # Performance par paire
        self.performance = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'last_trade': None
        })
        
        # Capital disponible
        self.available_capital = 10000.0  # USDT
        self.capital_per_pair = 2000.0    # Max par paire
        
        log_info(f"Gestionnaire multi-paires initialisé - Capital: {self.available_capital} USDT")
    
    async def initialize(self):
        """Initialise le gestionnaire"""
        try:
            # Scanner le marché pour la première fois
            await self.watchlist_scanner.update_watchlist()
            
            # Créer les stratégies pour chaque paire
            for symbol in self.watchlist_scanner.get_watchlist()[:5]:  # Top 5 pour commencer
                self.strategies[symbol] = MultiSignalStrategy(symbol=symbol)
                log_info(f"Stratégie créée pour {symbol}")
            
            return True
            
        except Exception as e:
            log_warning(f"Erreur lors de l'initialisation: {str(e)}")
            return False
    
    async def update_market_data(self):
        """Met à jour les données pour toutes les paires surveillées"""
        tasks = []
        
        for symbol in self.strategies.keys():
            # Créer une tâche async pour chaque paire
            task = self._update_pair_data(symbol)
            tasks.append(task)
        
        # Exécuter toutes les mises à jour en parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Logger les erreurs éventuelles
        for symbol, result in zip(self.strategies.keys(), results):
            if isinstance(result, Exception):
                log_warning(f"Erreur mise à jour {symbol}: {str(result)}")
    
    async def _update_pair_data(self, symbol: str):
        """Met à jour les données d'une paire spécifique"""
        try:
            # Récupérer les paramètres optimaux
            params = self.watchlist_scanner.get_trading_params(symbol)
            timeframe = params['timeframe']
            
            # Récupérer les klines
            symbol_ccxt = symbol.replace('/', '')
            response = self.exchange.publicGetKlines({
                'symbol': symbol_ccxt,
                'interval': timeframe,
                'limit': 200
            })
            
            # Convertir en format standard
            klines = []
            for candle in response:
                klines.append({
                    'timestamp': int(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            # Mettre à jour la stratégie
            self.strategies[symbol].update(klines)
            
        except Exception as e:
            raise Exception(f"Erreur update {symbol}: {str(e)}")
    
    async def check_signals(self):
        """Vérifie les signaux pour toutes les paires"""
        signals = {}
        
        for symbol, strategy in self.strategies.items():
            try:
                signal = strategy.get_signal()
                if signal:
                    signals[symbol] = {
                        'signal': signal,
                        'indicators': strategy.get_indicators(),
                        'params': self.watchlist_scanner.get_trading_params(symbol)
                    }
            except Exception as e:
                log_warning(f"Erreur signal {symbol}: {str(e)}")
        
        return signals
    
    def can_open_position(self, symbol: str) -> bool:
        """Vérifie si on peut ouvrir une position"""
        # Limite de positions simultanées
        if len(self.positions) >= self.config['trading']['max_positions']:
            return False
        
        # Position déjà ouverte sur cette paire
        if symbol in self.positions:
            return False
        
        # Capital disponible suffisant
        required_capital = self.capital_per_pair
        used_capital = sum(pos['capital'] for pos in self.positions.values())
        
        if self.available_capital - used_capital < required_capital:
            return False
        
        # Vérifier le cooldown (pas plus d'un trade par heure sur la même paire)
        perf = self.performance[symbol]
        if perf['last_trade']:
            time_since_last = (datetime.now() - perf['last_trade']).total_seconds()
            if time_since_last < 3600:  # 1 heure
                return False
        
        return True
    
    async def execute_signals(self, signals: dict):
        """Exécute les signaux de trading"""
        for symbol, signal_data in signals.items():
            signal = signal_data['signal']
            indicators = signal_data['indicators']
            params = signal_data['params']
            
            try:
                if signal == 'BUY' and self.can_open_position(symbol):
                    await self.open_position(symbol, indicators, params)
                    
                elif signal == 'SELL' and symbol in self.positions:
                    await self.close_position(symbol, indicators)
                    
            except Exception as e:
                log_warning(f"Erreur exécution {symbol}: {str(e)}")
    
    async def open_position(self, symbol: str, indicators: dict, params: dict):
        """Ouvre une position"""
        price = indicators['current_price']
        position_size_pct = params['position_size']
        
        # Calculer la taille en fonction du capital alloué
        capital_for_trade = min(self.capital_per_pair, self.available_capital * position_size_pct)
        size = capital_for_trade / price
        
        if self.paper_trading:
            # Simuler l'ouverture
            self.positions[symbol] = {
                'side': 'LONG',
                'entry_price': price,
                'size': size,
                'capital': capital_for_trade,
                'entry_time': datetime.now(),
                'stop_loss': price * (1 - params['stop_loss']),
                'take_profit': price * (1 + params['take_profit'])
            }
            
            # Mettre à jour la stratégie
            self.strategies[symbol].current_position = 'LONG'
            self.strategies[symbol].entry_price = price
            
            # Logger le trade
            log_trade(
                action='BUY',
                symbol=symbol,
                quantity=size,
                price=price,
                side='LONG',
                capital=capital_for_trade,
                buy_strength=indicators.get('buy_strength', 0),
                volatility=self.watchlist_scanner.get_pair_metrics(symbol).get('volatility', 0)
            )
            
            log_info(f"🟢 Position ouverte sur {symbol} - {size:.6f} @ {price:.2f} USDT")
    
    async def close_position(self, symbol: str, indicators: dict):
        """Ferme une position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = indicators['current_price']
        
        # Calculer le PnL
        pnl = (current_price - position['entry_price']) * position['size']
        pnl_pct = ((current_price / position['entry_price']) - 1) * 100
        
        if self.paper_trading:
            # Mettre à jour les performances
            perf = self.performance[symbol]
            perf['trades'] += 1
            perf['pnl'] += pnl
            perf['last_trade'] = datetime.now()
            
            if pnl > 0:
                perf['wins'] += 1
            else:
                perf['losses'] += 1
            
            # Logger le trade
            log_trade(
                action='SELL',
                symbol=symbol,
                quantity=position['size'],
                price=current_price,
                side='CLOSE',
                profit=pnl,
                pnl_pct=pnl_pct,
                sell_strength=indicators.get('sell_strength', 0)
            )
            
            emoji = "🟢" if pnl > 0 else "🔴"
            log_info(
                f"{emoji} Position fermée sur {symbol} - "
                f"PnL: {pnl:+.2f} USDT ({pnl_pct:+.2f}%)"
            )
            
            # Supprimer la position
            del self.positions[symbol]
            
            # Réinitialiser la stratégie
            self.strategies[symbol].current_position = None
            self.strategies[symbol].entry_price = None
    
    def get_performance_summary(self) -> dict:
        """Retourne un résumé des performances"""
        total_trades = sum(p['trades'] for p in self.performance.values())
        total_wins = sum(p['wins'] for p in self.performance.values())
        total_losses = sum(p['losses'] for p in self.performance.values())
        total_pnl = sum(p['pnl'] for p in self.performance.values())
        
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Performance par paire
        pair_perfs = []
        for symbol, perf in self.performance.items():
            if perf['trades'] > 0:
                pair_win_rate = (perf['wins'] / perf['trades'] * 100)
                pair_perfs.append({
                    'symbol': symbol,
                    'trades': perf['trades'],
                    'win_rate': pair_win_rate,
                    'pnl': perf['pnl']
                })
        
        # Trier par PnL
        pair_perfs.sort(key=lambda x: x['pnl'], reverse=True)
        
        return {
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'positions_open': len(self.positions),
            'pairs_trading': len(self.strategies),
            'top_performers': pair_perfs[:3],
            'worst_performers': pair_perfs[-3:] if len(pair_perfs) > 3 else []
        }
    
    async def update_watchlist(self):
        """Met à jour la watchlist et ajoute/retire des stratégies"""
        old_watchlist = set(self.strategies.keys())
        
        # Scanner le marché
        await self.watchlist_scanner.update_watchlist()
        
        new_watchlist = set(self.watchlist_scanner.get_watchlist()[:5])  # Top 5
        
        # Paires à ajouter
        to_add = new_watchlist - old_watchlist
        for symbol in to_add:
            if symbol not in self.positions:  # Ne pas ajouter si position ouverte
                self.strategies[symbol] = MultiSignalStrategy(symbol=symbol)
                log_info(f"➕ Nouvelle paire ajoutée: {symbol}")
        
        # Paires à retirer
        to_remove = old_watchlist - new_watchlist
        for symbol in to_remove:
            if symbol not in self.positions:  # Ne pas retirer si position ouverte
                del self.strategies[symbol]
                log_info(f"➖ Paire retirée: {symbol}")

    def get_positions(self) -> Dict:
        """
        Retourne les positions actuelles
        
        Returns:
            Dict des positions avec leurs détails
        """
        return {
            symbol: {
                'side': pos['side'],
                'size': pos['size'],
                'entry_price': pos['entry_price'],
                'current_price': pos['current_price'],
                'pnl': pos['pnl'],
                'pnl_pct': pos['pnl_pct'],
                'timestamp': pos['timestamp'].isoformat()
            }
            for symbol, pos in self.positions.items()
        }