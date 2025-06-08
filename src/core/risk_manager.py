"""
Module de gestion des risques pour le bot de trading
Gère les limites de position, drawdown, et protection du capital
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.core.logger import log_info, log_warning, log_error

@dataclass
class RiskMetrics:
    """Métriques de risque actuelles"""
    total_exposure: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float
    open_positions: int
    win_rate: float
    sharpe_ratio: float
    risk_score: float  # 0-100, 100 = risque maximal

class RiskManager:
    """
    Gestionnaire de risque complet pour le trading de cryptomonnaies
    """
    
    def __init__(self, config: dict):
        """
        Initialise le gestionnaire de risque
        
        Args:
            config: Configuration avec les paramètres de risque
        """
        # Paramètres de base
        self.max_position_size = config.get('max_position_size', 0.02)  # 2% par trade
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5% perte max/jour
        self.max_open_positions = config.get('max_open_positions', 3)
        self.max_drawdown = config.get('max_drawdown', 0.10)  # 10% drawdown max
        
        # Stop loss et take profit par défaut
        self.default_stop_loss = config.get('default_stop_loss', 0.05)  # 5%
        self.default_take_profit = config.get('default_take_profit', 0.10)  # 10%
        
        # Kelly Criterion
        self.use_kelly = config.get('use_kelly_criterion', True)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # 25% du Kelly
        
        # Trailing stop
        self.use_trailing_stop = config.get('use_trailing_stop', True)
        self.trailing_stop_distance = config.get('trailing_stop_distance', 0.03)  # 3%
        
        # État interne
        self.positions = {}  # Positions ouvertes
        self.daily_trades = []  # Trades du jour
        self.performance_history = []  # Historique des performances
        self.equity_curve = [10000.0]  # Capital initial
        self.peak_equity = 10000.0
        
        # Régimes de marché
        self.market_regime = 'NORMAL'  # NORMAL, VOLATILE, CRASH
        self.regime_params = {
            'NORMAL': {
                'position_size_mult': 1.0,
                'stop_loss_mult': 1.0,
                'max_positions': self.max_open_positions
            },
            'VOLATILE': {
                'position_size_mult': 0.5,
                'stop_loss_mult': 1.5,
                'max_positions': max(1, self.max_open_positions - 1)
            },
            'CRASH': {
                'position_size_mult': 0.0,  # Pas de nouvelles positions
                'stop_loss_mult': 2.0,
                'max_positions': 0
            }
        }
        
        log_info("RiskManager initialisé avec les paramètres:")
        log_info(f"  - Position max: {self.max_position_size*100:.1f}%")
        log_info(f"  - Perte quotidienne max: {self.max_daily_loss*100:.1f}%")
        log_info(f"  - Drawdown max: {self.max_drawdown*100:.1f}%")
    
    def can_open_position(self, symbol: str, capital: float) -> Tuple[bool, str]:
        """
        Vérifie si on peut ouvrir une nouvelle position
        
        Args:
            symbol: Symbole à trader
            capital: Capital disponible
            
        Returns:
            (peut_ouvrir, raison)
        """
        # Vérifier le régime de marché
        if self.market_regime == 'CRASH':
            return False, "Mode protection activé (crash détecté)"
        
        # Vérifier le nombre de positions
        regime = self.regime_params[self.market_regime]
        if len(self.positions) >= regime['max_positions']:
            return False, f"Limite de {regime['max_positions']} positions atteinte"
        
        # Vérifier si déjà une position sur ce symbole
        if symbol in self.positions:
            return False, f"Position déjà ouverte sur {symbol}"
        
        # Vérifier la perte quotidienne
        daily_loss = self.calculate_daily_loss()
        if daily_loss >= self.max_daily_loss:
            return False, f"Perte quotidienne limite atteinte ({daily_loss*100:.1f}%)"
        
        # Vérifier le drawdown
        current_drawdown = self.calculate_current_drawdown()
        if current_drawdown >= self.max_drawdown:
            return False, f"Drawdown limite atteint ({current_drawdown*100:.1f}%)"
        
        # Vérifier l'exposition totale
        total_exposure = self.calculate_total_exposure(capital)
        max_allowed = 0.5  # Max 50% du capital en positions
        if total_exposure >= max_allowed:
            return False, f"Exposition totale trop élevée ({total_exposure*100:.1f}%)"
        
        return True, "OK"
    
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                              capital: float, current_price: float) -> float:
        """
        Calcule la taille de position optimale
        
        Args:
            symbol: Symbole à trader
            signal_strength: Force du signal (0-1)
            capital: Capital disponible
            current_price: Prix actuel
            
        Returns:
            Taille de position en unités
        """
        # Position de base
        base_position_value = capital * self.max_position_size
        
        # Ajuster selon le régime de marché
        regime = self.regime_params[self.market_regime]
        base_position_value *= regime['position_size_mult']
        
        # Ajuster selon la force du signal
        base_position_value *= signal_strength
        
        # Kelly Criterion si activé
        if self.use_kelly:
            kelly_size = self._calculate_kelly_size(symbol, capital)
            base_position_value = min(base_position_value, kelly_size)
        
        # Convertir en unités
        position_size = base_position_value / current_price
        
        # Vérifier les limites
        min_position_value = 50  # Minimum 50 USDT
        if base_position_value < min_position_value:
            return 0.0
        
        return position_size
    
    def _calculate_kelly_size(self, symbol: str, capital: float) -> float:
        """
        Calcule la taille selon le critère de Kelly
        
        Args:
            symbol: Symbole
            capital: Capital disponible
            
        Returns:
            Taille de position en valeur
        """
        # Calculer le win rate et le ratio gain/perte
        stats = self._get_symbol_stats(symbol)
        
        if stats['total_trades'] < 10:
            # Pas assez de données, utiliser la taille par défaut
            return capital * self.max_position_size * 0.5
        
        win_rate = stats['win_rate']
        avg_win = stats['avg_win']
        avg_loss = abs(stats['avg_loss'])
        
        if avg_loss == 0:
            return capital * self.max_position_size
        
        # Formule de Kelly: f = (p * b - q) / b
        # où p = probabilité de gain, q = 1-p, b = ratio gain/perte
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b
        
        # Limiter et appliquer la fraction de Kelly
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25%
        kelly_fraction *= self.kelly_fraction  # Appliquer la fraction conservative
        
        return capital * kelly_fraction
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, 
                          side: str = 'LONG') -> float:
        """
        Calcule le niveau de stop loss
        
        Args:
            symbol: Symbole
            entry_price: Prix d'entrée
            side: Direction (LONG/SHORT)
            
        Returns:
            Prix de stop loss
        """
        # Stop loss de base
        stop_distance = self.default_stop_loss
        
        # Ajuster selon le régime
        regime = self.regime_params[self.market_regime]
        stop_distance *= regime['stop_loss_mult']
        
        # Ajuster selon la volatilité
        volatility = self._get_symbol_volatility(symbol)
        if volatility > 0.05:  # Volatilité > 5%
            stop_distance *= 1.5
        
        # Calculer le prix
        if side == 'LONG':
            stop_price = entry_price * (1 - stop_distance)
        else:
            stop_price = entry_price * (1 + stop_distance)
        
        return stop_price
    
    def calculate_take_profit(self, symbol: str, entry_price: float, 
                            side: str = 'LONG') -> float:
        """
        Calcule le niveau de take profit
        
        Args:
            symbol: Symbole
            entry_price: Prix d'entrée
            side: Direction
            
        Returns:
            Prix de take profit
        """
        # Ratio risk/reward minimum de 2:1
        stop_distance = self.default_stop_loss
        tp_distance = stop_distance * 2
        
        # Ajuster selon les stats du symbole
        stats = self._get_symbol_stats(symbol)
        if stats['avg_win'] > 0 and stats['total_trades'] > 10:
            # Utiliser le gain moyen historique
            tp_distance = stats['avg_win'] * 1.2
        
        # Calculer le prix
        if side == 'LONG':
            tp_price = entry_price * (1 + tp_distance)
        else:
            tp_price = entry_price * (1 - tp_distance)
        
        return tp_price
    
    def update_trailing_stop(self, position: dict, current_price: float) -> Optional[float]:
        """
        Met à jour le trailing stop
        
        Args:
            position: Position actuelle
            current_price: Prix actuel
            
        Returns:
            Nouveau stop loss ou None
        """
        if not self.use_trailing_stop:
            return None
        
        side = position['side']
        current_stop = position.get('stop_loss', 0)
        
        if side == 'LONG':
            # Le prix a monté, on remonte le stop
            new_stop = current_price * (1 - self.trailing_stop_distance)
            if new_stop > current_stop:
                return new_stop
        else:  # SHORT
            # Le prix a baissé, on baisse le stop
            new_stop = current_price * (1 + self.trailing_stop_distance)
            if new_stop < current_stop:
                return new_stop
        
        return None
    
    def detect_market_regime(self, market_data: dict):
        """
        Détecte le régime de marché actuel
        
        Args:
            market_data: Données de marché globales
        """
        # Indicateurs de régime
        btc_change_24h = market_data.get('btc_change_24h', 0)
        market_fear_greed = market_data.get('fear_greed_index', 50)
        volatility_index = market_data.get('volatility_index', 0)
        
        old_regime = self.market_regime
        
        # Détection de crash
        if btc_change_24h < -10 or market_fear_greed < 20:
            self.market_regime = 'CRASH'
        # Marché volatil
        elif volatility_index > 0.05 or abs(btc_change_24h) > 5:
            self.market_regime = 'VOLATILE'
        # Marché normal
        else:
            self.market_regime = 'NORMAL'
        
        if old_regime != self.market_regime:
            log_warning(f"Changement de régime: {old_regime} → {self.market_regime}")
            
            # Actions selon le nouveau régime
            if self.market_regime == 'CRASH':
                log_error("🚨 MODE CRASH ACTIVÉ - Protection du capital")
            elif self.market_regime == 'VOLATILE':
                log_warning("⚠️ Marché volatil - Réduction des positions")
    
    def add_position(self, position: dict):
        """Ajoute une position au tracking"""
        symbol = position['symbol']
        self.positions[symbol] = {
            **position,
            'entry_time': datetime.now(),
            'highest_price': position['entry_price'],
            'lowest_price': position['entry_price']
        }
        
        # Ajouter au journal quotidien
        self.daily_trades.append({
            'symbol': symbol,
            'action': 'OPEN',
            'time': datetime.now(),
            'price': position['entry_price'],
            'size': position['size']
        })
    
    def close_position(self, symbol: str, exit_price: float) -> dict:
        """
        Ferme une position et calcule les métriques
        
        Args:
            symbol: Symbole
            exit_price: Prix de sortie
            
        Returns:
            Résultat du trade
        """
        if symbol not in self.positions:
            return {}
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        # Calculer le PnL
        if side == 'LONG':
            pnl = (exit_price - entry_price) * size
            pnl_pct = ((exit_price / entry_price) - 1) * 100
        else:
            pnl = (entry_price - exit_price) * size
            pnl_pct = ((entry_price / exit_price) - 1) * 100
        
        # Durée du trade
        duration = datetime.now() - position['entry_time']
        
        # Mettre à jour les stats
        trade_result = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration': duration,
            'win': pnl > 0
        }
        
        # Ajouter à l'historique
        self.performance_history.append(trade_result)
        
        # Mettre à jour l'equity curve
        current_equity = self.equity_curve[-1] + pnl
        self.equity_curve.append(current_equity)
        
        # Mettre à jour le peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Supprimer la position
        del self.positions[symbol]
        
        # Logger
        self.daily_trades.append({
            'symbol': symbol,
            'action': 'CLOSE',
            'time': datetime.now(),
            'price': exit_price,
            'pnl': pnl
        })
        
        return trade_result
    
    def calculate_daily_loss(self) -> float:
        """Calcule la perte du jour en pourcentage"""
        today_trades = [
            t for t in self.daily_trades 
            if t['time'].date() == datetime.now().date() and 'pnl' in t
        ]
        
        if not today_trades:
            return 0.0
        
        daily_pnl = sum(t.get('pnl', 0) for t in today_trades)
        start_equity = self.equity_curve[-len(today_trades)-1] if len(self.equity_curve) > len(today_trades) else self.equity_curve[0]
        
        return -daily_pnl / start_equity if daily_pnl < 0 else 0.0
    
    def calculate_current_drawdown(self) -> float:
        """Calcule le drawdown actuel"""
        current_equity = self.equity_curve[-1]
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        return max(0, drawdown)
    
    def calculate_total_exposure(self, capital: float) -> float:
        """Calcule l'exposition totale en pourcentage du capital"""
        total_value = sum(
            pos['size'] * pos['entry_price'] 
            for pos in self.positions.values()
        )
        return total_value / capital if capital > 0 else 0
    
    def get_risk_metrics(self, capital: float) -> RiskMetrics:
        """
        Calcule toutes les métriques de risque actuelles
        
        Args:
            capital: Capital total
            
        Returns:
            RiskMetrics avec toutes les métriques
        """
        # Métriques de base
        total_exposure = self.calculate_total_exposure(capital)
        daily_pnl = self.calculate_daily_loss()
        current_drawdown = self.calculate_current_drawdown()
        
        # Calculer le win rate
        total_trades = len(self.performance_history)
        wins = sum(1 for t in self.performance_history if t['win'])
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # Calculer le Sharpe ratio (simplifié)
        if len(self.equity_curve) > 30:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        # Score de risque global (0-100)
        risk_score = (
            (current_drawdown / self.max_drawdown) * 40 +
            (daily_pnl / self.max_daily_loss) * 30 +
            (total_exposure / 0.5) * 20 +
            (1 - win_rate) * 10
        ) * 100
        
        return RiskMetrics(
            total_exposure=total_exposure,
            daily_pnl=daily_pnl,
            max_drawdown=self.max_drawdown,
            current_drawdown=current_drawdown,
            open_positions=len(self.positions),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            risk_score=min(100, risk_score)
        )
    
    def _get_symbol_stats(self, symbol: str) -> dict:
        """Récupère les statistiques pour un symbole"""
        symbol_trades = [
            t for t in self.performance_history 
            if t['symbol'] == symbol
        ]
        
        if not symbol_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.5,
                'avg_win': 0.02,
                'avg_loss': -0.01
            }
        
        wins = [t for t in symbol_trades if t['win']]
        losses = [t for t in symbol_trades if not t['win']]
        
        return {
            'total_trades': len(symbol_trades),
            'win_rate': len(wins) / len(symbol_trades),
            'avg_win': np.mean([t['pnl_pct'] for t in wins]) / 100 if wins else 0.02,
            'avg_loss': np.mean([t['pnl_pct'] for t in losses]) / 100 if losses else -0.01
        }
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Estime la volatilité d'un symbole"""
        # TODO: Implémenter avec les données réelles
        # Pour l'instant, retourner une valeur par défaut
        return 0.03  # 3% de volatilité
    
    def reset_daily_counters(self):
        """Réinitialise les compteurs quotidiens"""
        # Garder seulement les trades d'aujourd'hui
        today = datetime.now().date()
        self.daily_trades = [
            t for t in self.daily_trades 
            if t['time'].date() == today
        ]
        
        log_info("Compteurs quotidiens réinitialisés")
    
    def emergency_close_all(self, reason: str = "Protection d'urgence"):
        """
        Ferme toutes les positions en urgence
        
        Args:
            reason: Raison de la fermeture
        """
        log_error(f"🚨 FERMETURE D'URGENCE: {reason}")
        
        positions_to_close = list(self.positions.keys())
        for symbol in positions_to_close:
            # Simuler une fermeture au prix actuel
            # Dans la vraie implémentation, récupérer le prix réel
            current_price = self.positions[symbol]['entry_price'] * 0.95  # -5% pour simuler
            self.close_position(symbol, current_price)
        
        log_warning(f"Fermé {len(positions_to_close)} positions")