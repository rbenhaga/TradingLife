"""
Module de configuration pour le bot de trading
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier JSON
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Charger les variables d'environnement si les clés API ne sont pas dans le fichier
        if not config['exchange']['api_key']:
            config['exchange']['api_key'] = os.getenv('BINANCE_API_KEY', '')
        if not config['exchange']['api_secret']:
            config['exchange']['api_secret'] = os.getenv('BINANCE_API_SECRET', '')
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Erreur lors de la lecture du fichier JSON: {config_path}")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Valide la configuration
    
    Args:
        config: Dictionnaire de configuration
        
    Returns:
        True si la configuration est valide
    """
    required_keys = {
        'exchange': ['name', 'testnet', 'api_key', 'api_secret'],
        'trading': ['pairs', 'initial_balance', 'position_size', 'max_positions'],
        'risk_management': ['max_drawdown', 'daily_loss_limit'],
        'logging': ['level', 'file']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Section manquante dans la configuration: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Clé manquante dans la configuration: {section}.{key}")
    # Spécial pour strategy: accepter timeframe OU timeframes
    strat = config.get('strategy', {})
    for key in ['name', 'short_window', 'long_window']:
        if key not in strat:
            raise ValueError(f"Clé manquante dans la configuration: strategy.{key}")
    if 'timeframe' not in strat and 'timeframes' not in strat:
        raise ValueError("Clé manquante dans la configuration: strategy.timeframe OU strategy.timeframes")
    # Valider les valeurs
    if strat['short_window'] >= strat['long_window']:
        raise ValueError("La fenêtre courte doit être inférieure à la fenêtre longue")
    if config['trading']['position_size'] > 0.1:
        raise ValueError("La taille de position ne doit pas dépasser 10%")
    if not config['exchange']['api_key'] or not config['exchange']['api_secret']:
        raise ValueError("Les clés API doivent être configurées")
    return True

# Configuration par défaut pour les tests
default_config = {
    "exchange": {
        "name": "binance",
        "testnet": True,
        "api_key": os.getenv('BINANCE_API_KEY', ''),
        "api_secret": os.getenv('BINANCE_API_SECRET', '')
    },
    "trading": {
        "pairs": ["BTC/USDT"],
        "initial_balance": 10000,
        "position_size": 0.02,
        "max_positions": 1,
        "stop_loss": 0.02,
        "take_profit": 0.05
    },
    "strategy": {
        "name": "ma_cross",
        "short_window": 5,
        "long_window": 13,
        "trend_window": 50,  # Réduit de 200 à 50
        "timeframe": "1m",
        "min_data_points": 100
    },
    "risk_management": {
        "max_drawdown": 0.1,
        "daily_loss_limit": 0.05,
        "position_sizing": "fixed"
    },
    "logging": {
        "level": "INFO",
        "file": "logs/trading.log"
    }
}