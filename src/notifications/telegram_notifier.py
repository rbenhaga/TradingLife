"""
Système de notifications Telegram simplifié
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

from ..core.logger import log_info, log_error


class NotificationLevel(Enum):
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    TRADE = "💰"
    ANALYSIS = "📊"


class TelegramNotifier:
    """Gestionnaire de notifications Telegram"""
    
    def __init__(self, bot_token: str, chat_ids: List[str]):
        self.bot_token = bot_token
        self.chat_ids = chat_ids
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.session = None
        
    async def initialize(self):
        """Initialise la session HTTP"""
        self.session = aiohttp.ClientSession()
        # Test de connexion
        try:
            await self.send_message("🚀 Bot de trading connecté!", NotificationLevel.INFO)
            return True
        except Exception as e:
            log_error(f"Erreur init Telegram: {e}")
            return False
    
    async def send_message(self, text: str, level: NotificationLevel = NotificationLevel.INFO):
        """Envoie un message formaté"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        formatted_text = f"{level.value} {text}"
        
        for chat_id in self.chat_ids:
            try:
                url = f"{self.api_url}/sendMessage"
                data = {
                    'chat_id': chat_id,
                    'text': formatted_text,
                    'parse_mode': 'HTML'
                }
                
                async with self.session.post(url, json=data) as response:
                    if response.status != 200:
                        log_error(f"Erreur Telegram: {await response.text()}")
                        
            except Exception as e:
                log_error(f"Erreur envoi message: {e}")
    
    async def notify_trade(self, trade: Dict):
        """Notifie un trade"""
        emoji = "🟢" if trade['side'] == 'BUY' else "🔴"
        message = f"""
<b>Trade Exécuté</b>
{emoji} {trade['symbol']} - {trade['side']}
💵 Prix: ${trade['price']:.2f}
📊 Quantité: {trade['quantity']:.6f}
🎯 Confiance: {trade.get('confidence', 0):.1%}
📝 Raison: {trade.get('reason', 'N/A')}
"""
        await self.send_message(message, NotificationLevel.TRADE)
    
    async def notify_daily_summary(self, summary: Dict):
        """Envoie le résumé quotidien"""
        message = f"""
<b>📈 Résumé Quotidien</b>

💰 <b>Capital:</b> ${summary['capital']:.2f}
📊 <b>P&L Jour:</b> ${summary['daily_pnl']:.2f} ({summary['daily_pnl_pct']:+.2f}%)
📈 <b>P&L Total:</b> ${summary['total_pnl']:.2f} ({summary['total_pnl_pct']:+.2f}%)

🎯 <b>Trades:</b> {summary['total_trades']} ({summary['wins']}W/{summary['losses']}L)
📊 <b>Win Rate:</b> {summary['win_rate']:.1f}%
📉 <b>Max Drawdown:</b> {summary['max_drawdown']:.1f}%
⚡ <b>Sharpe Ratio:</b> {summary['sharpe_ratio']:.2f}
"""
        await self.send_message(message, NotificationLevel.ANALYSIS)
    
    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()