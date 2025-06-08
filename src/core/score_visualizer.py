"""
Visualiseur de score pour comprendre les décisions de trading
Génère des représentations visuelles des scores et signaux
"""

from typing import Dict, List
import json
from datetime import datetime
from src.core.weighted_score_engine import TradingScore
from src.logger import log_info

class ScoreVisualizer:
    """Visualise les scores de trading pour une meilleure compréhension"""
    
    @staticmethod
    def create_score_bar(score: float, width: int = 20) -> str:
        """Crée une barre de progression ASCII pour le score"""
        # Score entre -1 et 1
        normalized = (score + 1) / 2  # Convertir en 0-1
        filled = int(normalized * width)
        
        # Couleurs ASCII
        if score > 0.5:
            color = "🟩"  # Vert fort
        elif score > 0.3:
            color = "🟢"  # Vert
        elif score > -0.3:
            color = "🟡"  # Jaune
        elif score > -0.5:
            color = "🟠"  # Orange
        else:
            color = "🔴"  # Rouge
        
        bar = f"{color} ["
        bar += "=" * filled
        bar += " " * (width - filled)
        bar += f"] {score:+.3f}"
        
        return bar
    
    @staticmethod
    def create_signal_breakdown(trading_score: TradingScore) -> str:
        """Crée une visualisation détaillée des signaux"""
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"📊 ANALYSE DE SCORE - {trading_score.timestamp.strftime('%H:%M:%S')}")
        output.append(f"{'='*60}")
        
        # Score total avec barre
        output.append(f"\n🎯 Score Total: {ScoreVisualizer.create_score_bar(trading_score.total_score)}")
        output.append(f"📈 Direction: {trading_score.direction}")
        output.append(f"💪 Confiance: {trading_score.confidence:.0%}")
        
        # Breakdown par signal
        output.append(f"\n{'─'*60}")
        output.append("📋 Détail des Signaux:")
        output.append(f"{'─'*60}")
        
        # Trier par contribution absolue
        sorted_signals = sorted(trading_score.signals, 
                              key=lambda s: abs(s.weighted_value), 
                              reverse=True)
        
        for signal in sorted_signals:
            # Icône selon le signal
            icons = {
                'rsi': '📉',
                'bollinger': '📊',
                'macd': '📈',
                'volume': '📢',
                'ma_cross': '✂️',
                'momentum': '🚀',
                'volatility': '📍'
            }
            icon = icons.get(signal.name, '📌')
            
            # Barre de contribution
            contrib_bar = ScoreVisualizer.create_contribution_bar(signal.weighted_value)
            
            output.append(
                f"{icon} {signal.name.upper():10} {contrib_bar} | "
                f"Poids: {signal.weight:.0%} | "
                f"Conf: {signal.confidence:.0%}"
            )
            output.append(f"   → {signal.reason}")
            output.append("")
        
        # Recommandation
        output.append(f"{'─'*60}")
        output.append("💡 Recommandation:")
        
        if trading_score.direction == 'STRONG_BUY':
            output.append("   ✅ ACHETER MAINTENANT - Signal très fort!")
        elif trading_score.direction == 'BUY':
            output.append("   ✅ Acheter - Bon signal d'entrée")
        elif trading_score.direction == 'STRONG_SELL':
            output.append("   ❌ VENDRE MAINTENANT - Signal très fort!")
        elif trading_score.direction == 'SELL':
            output.append("   ❌ Vendre - Signal de sortie")
        else:
            output.append("   ⏸️ Attendre - Pas de signal clair")
        
        output.append(f"{'='*60}\n")
        
        return "\n".join(output)
    
    @staticmethod
    def create_contribution_bar(value: float, width: int = 15) -> str:
        """Crée une barre pour montrer la contribution d'un signal"""
        abs_value = abs(value)
        filled = int(abs_value * width * 2)  # *2 car value max = 0.5
        
        if value > 0:
            # Contribution positive (achat)
            bar = "+" + "▰" * min(filled, width)
            bar = bar.ljust(width + 1)
            return f"[{bar}]"
        else:
            # Contribution négative (vente)
            bar = "-" + "▰" * min(filled, width)
            bar = bar.ljust(width + 1)
            return f"[{bar}]"
    
    @staticmethod
    def create_mini_summary(trading_score: TradingScore) -> str:
        """Crée un résumé court sur une ligne"""
        # Emoji selon la direction
        emojis = {
            'STRONG_BUY': '🟢🟢',
            'BUY': '🟢',
            'NEUTRAL': '⚪',
            'SELL': '🔴',
            'STRONG_SELL': '🔴🔴'
        }
        emoji = emojis.get(trading_score.direction, '❓')
        
        # Top contributeur
        top_signal = max(trading_score.signals, key=lambda s: abs(s.weighted_value))
        
        return f"{emoji} Score: {trading_score.total_score:+.3f} | {top_signal.name}: {top_signal.reason}"
    
    @staticmethod
    def create_comparison_table(scores: Dict[str, TradingScore]) -> str:
        """Compare les scores de plusieurs paires"""
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"📊 COMPARAISON MULTI-PAIRES - {datetime.now().strftime('%H:%M:%S')}")
        output.append(f"{'='*80}")
        output.append(f"{'Paire':10} {'Score':>10} {'Direction':>15} {'Confiance':>12} {'Top Signal':>25}")
        output.append(f"{'-'*80}")
        
        # Trier par score absolu
        sorted_pairs = sorted(scores.items(), 
                            key=lambda x: abs(x[1].total_score), 
                            reverse=True)
        
        for symbol, score in sorted_pairs:
            # Top signal
            top_signal = max(score.signals, key=lambda s: abs(s.weighted_value))
            
            # Couleur selon la direction
            if score.direction in ['STRONG_BUY', 'BUY']:
                direction_str = f"🟢 {score.direction}"
            elif score.direction in ['STRONG_SELL', 'SELL']:
                direction_str = f"🔴 {score.direction}"
            else:
                direction_str = f"⚪ {score.direction}"
            
            output.append(
                f"{symbol:10} {score.total_score:+10.3f} {direction_str:>15} "
                f"{score.confidence:>11.0%} {top_signal.name:>25}"
            )
        
        output.append(f"{'='*80}\n")
        
        return "\n".join(output)
    
    @staticmethod
    def log_score_analysis(symbol: str, trading_score: TradingScore):
        """Log une analyse complète du score"""
        if trading_score.direction != 'NEUTRAL':
            analysis = ScoreVisualizer.create_signal_breakdown(trading_score)
            log_info(f"{symbol} - Analyse de décision:\n{analysis}")
        else:
            summary = ScoreVisualizer.create_mini_summary(trading_score)
            log_info(f"{symbol} - {summary}")