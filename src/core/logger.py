"""
Module de logging pour le bot de trading
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
import json

# Configuration du logger
logger = logging.getLogger('TradingBot')
logger.setLevel(logging.DEBUG)

# S'assurer que le logger n'a pas de handlers dupliqués
if not logger.handlers:
    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Format des logs avec couleurs
    class ColoredFormatter(logging.Formatter):
        """Formatter qui ajoute des couleurs aux logs"""
        
        COLORS = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Vert
            'WARNING': '\033[33m',  # Jaune
            'ERROR': '\033[31m',    # Rouge
            'CRITICAL': '\033[35m', # Magenta
        }
        RESET = '\033[0m'
        
        def format(self, record):
            log_color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{log_color}{record.levelname}{self.RESET}"
            return super().format(record)
    
    # Utiliser le formatter coloré
    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Ajout du handler au logger
    logger.addHandler(console_handler)

def log_info(message: str, **kwargs):
    """Log un message de niveau INFO"""
    if kwargs:
        message = f"{message} | {json.dumps(kwargs)}"
    logger.info(message)

def log_error(message: str, **kwargs):
    """Log un message de niveau ERROR"""
    if kwargs:
        message = f"{message} | {json.dumps(kwargs)}"
    logger.error(message)

def log_debug(message: str, **kwargs):
    """Log un message de niveau DEBUG"""
    if kwargs:
        message = f"{message} | {json.dumps(kwargs)}"
    logger.debug(message)

def log_warning(message: str, **kwargs):
    """Log un message de niveau WARNING"""
    if kwargs:
        message = f"{message} | {json.dumps(kwargs)}"
    logger.warning(message)

def log_critical(message: str, **kwargs):
    """Log un message de niveau CRITICAL"""
    if kwargs:
        message = f"{message} | {json.dumps(kwargs)}"
    logger.critical(message)

# Fonction pour logger les trades
def log_trade(action: str, symbol: str, quantity: float, price: float, 
              side: str, profit: float | None = None, **kwargs):
    """
    Log une opération de trading
    
    Args:
        action: Type d'action (BUY, SELL, etc.)
        symbol: Symbole tradé
        quantity: Quantité
        price: Prix d'exécution
        side: Côté (LONG, SHORT)
        profit: Profit/perte si applicable
        **kwargs: Autres informations
    """
    trade_info = {
        'action': action,
        'symbol': symbol,
        'quantity': quantity,
        'price': price,
        'side': side,
        'timestamp': datetime.now().isoformat()
    }
    
    if profit is not None:
        trade_info['profit'] = profit
        
    trade_info.update(kwargs)
    
    # Formater le message
    if action == 'BUY':
        emoji = '🟢'
    elif action == 'SELL':
        emoji = '🔴'
    else:
        emoji = '📊'
        
    message = f"{emoji} TRADE: {action} {quantity:.6f} {symbol} @ {price:.2f} USDT"
    
    if profit is not None:
        if profit > 0:
            message += f" | Profit: +{profit:.2f} USDT ✅"
        else:
            message += f" | Loss: {profit:.2f} USDT ❌"
    
    logger.info(message)
    
    # Sauvegarder dans un fichier de trades séparé
    try:
        trades_file = Path("data/logs/trades.jsonl")
        trades_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(trades_file, 'a') as f:
            f.write(json.dumps(trade_info) + '\n')
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du trade: {e}")

# Fonction pour logger les performances
def log_performance(metrics: dict):
    """
    Log les métriques de performance
    
    Args:
        metrics: Dictionnaire des métriques
    """
    message_parts = ["📊 PERFORMANCE:"]
    
    if 'total_trades' in metrics:
        message_parts.append(f"Trades: {metrics['total_trades']}")
    
    if 'win_rate' in metrics:
        message_parts.append(f"Win Rate: {metrics['win_rate']:.1f}%")
    
    if 'total_pnl' in metrics:
        pnl = metrics['total_pnl']
        if pnl >= 0:
            message_parts.append(f"PnL: +{pnl:.2f} USDT")
        else:
            message_parts.append(f"PnL: {pnl:.2f} USDT")
    
    if 'current_drawdown' in metrics:
        message_parts.append(f"Drawdown: {metrics['current_drawdown']:.2f}%")
    
    logger.info(" | ".join(message_parts))