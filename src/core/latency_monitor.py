# src/core/latency_monitor.py
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
from .logger import log_warning

class LatencyMonitor:
    def __init__(self, window_size=100, alert_threshold=250):
        self.latencies = defaultdict(lambda: deque(maxlen=window_size))
        self.alert_counts = defaultdict(int)
        self.last_alert = defaultdict(lambda: datetime.min)
        self.alert_threshold = alert_threshold
        self.alert_cooldown = timedelta(minutes=1)
        
    def add_latency(self, symbol: str, latency_ms: float):
        """Ajoute une mesure de latence"""
        self.latencies[symbol].append(latency_ms)
        
        # Vérifier si on doit alerter
        if latency_ms > self.alert_threshold:
            self.alert_counts[symbol] += 1
            
            # Ne pas spammer les logs
            if datetime.now() - self.last_alert[symbol] > self.alert_cooldown:
                avg_latency = statistics.mean(self.latencies[symbol])
                max_latency = max(self.latencies[symbol])
                
                log_warning(
                    f"⚠️ Latence élevée sur {symbol}: "
                    f"Moy={avg_latency:.1f}ms, Max={max_latency:.1f}ms "
                    f"({self.alert_counts[symbol]} alertes)"
                )
                self.last_alert[symbol] = datetime.now()
                self.alert_counts[symbol] = 0  # Reset counter
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de latence"""
        stats = {}
        for symbol, values in self.latencies.items():
            if values:
                stats[symbol] = {
                    'avg': statistics.mean(values),
                    'max': max(values),
                    'min': min(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) > 20 else max(values)
                }
        return stats