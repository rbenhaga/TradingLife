# src/monitoring/prometheus_metrics.py
"""
Métriques Prometheus pour monitoring temps réel
Expose les KPIs critiques du bot
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client import start_http_server, CollectorRegistry
import time
from functools import wraps
from typing import Dict, Any
import psutil
import os

# Registre personnalisé pour éviter les conflits
registry = CollectorRegistry()

# ===== Métriques de Trading =====

# Compteurs
trades_total = Counter(
    'trading_bot_trades_total',
    'Nombre total de trades exécutés',
    ['symbol', 'side', 'strategy'],
    registry=registry
)

orders_placed = Counter(
    'trading_bot_orders_placed_total',
    'Nombre total d\'ordres placés',
    ['symbol', 'order_type', 'status'],
    registry=registry
)

errors_total = Counter(
    'trading_bot_errors_total',
    'Nombre total d\'erreurs',
    ['component', 'error_type'],
    registry=registry
)

# Histogrammes (distributions)
trade_latency = Histogram(
    'trading_bot_trade_latency_seconds',
    'Latence d\'exécution des trades',
    ['exchange'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry
)

decision_latency = Histogram(
    'trading_bot_decision_latency_seconds',
    'Latence de prise de décision',
    ['strategy'],
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
    registry=registry
)

pnl_distribution = Histogram(
    'trading_bot_pnl_distribution_usdt',
    'Distribution des P&L par trade',
    ['symbol', 'strategy'],
    buckets=(-100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 200, 500),
    registry=registry
)

# Jauges (valeurs instantanées)
open_positions = Gauge(
    'trading_bot_open_positions',
    'Nombre de positions ouvertes',
    ['symbol'],
    registry=registry
)

account_balance = Gauge(
    'trading_bot_account_balance_usdt',
    'Balance du compte en USDT',
    ['asset'],
    registry=registry
)

current_drawdown = Gauge(
    'trading_bot_current_drawdown_percent',
    'Drawdown actuel en pourcentage',
    registry=registry
)

sharpe_ratio = Gauge(
    'trading_bot_sharpe_ratio',
    'Sharpe ratio actuel',
    ['timeframe'],
    registry=registry
)

# ===== Métriques Système =====

cpu_usage = Gauge(
    'trading_bot_cpu_usage_percent',
    'Utilisation CPU du bot',
    registry=registry
)

memory_usage = Gauge(
    'trading_bot_memory_usage_mb',
    'Utilisation mémoire en MB',
    registry=registry
)

websocket_connected = Gauge(
    'trading_bot_websocket_connected',
    'État de connexion WebSocket (1=connecté, 0=déconnecté)',
    ['exchange'],
    registry=registry
)

market_data_lag = Histogram(
    'trading_bot_market_data_lag_ms',
    'Lag des données de marché en millisecondes',
    ['symbol'],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
    registry=registry
)

# ===== Métriques Stratégie =====

strategy_signals = Counter(
    'trading_bot_strategy_signals_total',
    'Nombre de signaux générés',
    ['strategy', 'signal_type', 'symbol'],
    registry=registry
)

strategy_confidence = Histogram(
    'trading_bot_strategy_confidence',
    'Distribution de la confiance des signaux',
    ['strategy'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry
)

# Info sur le bot
bot_info = Info(
    'trading_bot_info',
    'Informations sur le bot',
    registry=registry
)

# ===== Décorateurs pour mesurer =====

def measure_latency(metric: Histogram, labels: Dict[str, str] | None = None):
    """Décorateur pour mesurer la latence d'une fonction"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator

def count_errors(component: str):
    """Décorateur pour compter les erreurs"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = type(e).__name__
                errors_total.labels(
                    component=component,
                    error_type=error_type
                ).inc()
                raise
        return wrapper
    return decorator

# ===== Collecteur de métriques système =====

class SystemMetricsCollector:
    """Collecte les métriques système périodiquement"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def collect(self):
        """Met à jour les métriques système"""
        # CPU
        cpu_percent = self.process.cpu_percent(interval=0.1)
        cpu_usage.set(cpu_percent)
        
        # Mémoire
        mem_info = self.process.memory_info()
        memory_mb = mem_info.rss / 1024 / 1024
        memory_usage.set(memory_mb)

# ===== Serveur Prometheus =====

class PrometheusExporter:
    """Exporte les métriques vers Prometheus"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.system_collector = SystemMetricsCollector()
        self._running = False
    
    def start(self):
        """Démarre le serveur de métriques"""
        start_http_server(self.port, registry=registry)
        self._running = True
        
        # Informations du bot
        bot_info.info({
            'version': '3.0',
            'strategy': 'scalping',
            'exchange': 'binance'
        })
        
        print(f"✅ Prometheus metrics server started on port {self.port}")
    
    def update_trading_metrics(self, metrics: Dict[str, Any]):
        """Met à jour les métriques de trading"""
        # Positions
        if 'positions' in metrics:
            for symbol, count in metrics['positions'].items():
                open_positions.labels(symbol=symbol).set(count)
        
        # Balance
        if 'balance' in metrics:
            for asset, amount in metrics['balance'].items():
                account_balance.labels(asset=asset).set(amount)
        
        # Performance
        if 'drawdown' in metrics:
            current_drawdown.set(metrics['drawdown'])
        
        if 'sharpe' in metrics:
            sharpe_ratio.labels(timeframe='daily').set(metrics['sharpe'])
    
    def record_trade(self, symbol: str, side: str, pnl: float, 
                     strategy: str = 'scalping'):
        """Enregistre un trade"""
        trades_total.labels(
            symbol=symbol,
            side=side,
            strategy=strategy
        ).inc()
        
        pnl_distribution.labels(
            symbol=symbol,
            strategy=strategy
        ).observe(pnl)
    
    def record_signal(self, strategy: str, signal_type: str, 
                      symbol: str, confidence: float):
        """Enregistre un signal de stratégie"""
        strategy_signals.labels(
            strategy=strategy,
            signal_type=signal_type,
            symbol=symbol
        ).inc()
        
        strategy_confidence.labels(
            strategy=strategy
        ).observe(confidence)
    
    def update_system_metrics(self):
        """Met à jour les métriques système"""
        self.system_collector.collect()