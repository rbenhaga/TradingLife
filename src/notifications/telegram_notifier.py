"""
Système de notifications Telegram simplifié
"""

import asyncio
import aiohttp
import os
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

from ..core.logger import log_info, log_error


class NotificationLevel(Enum):
    INFO = "ℹ️"
    WARNING = "⚠️"
    ALERT = "🚨"
    SUCCESS = "✅"
    TRADE = "📊"


class TelegramNotifier:
    """Gestionnaire de notifications Telegram"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            print("⚠️ Telegram non configuré - Ajoutez TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID au .env")
    
    async def send_message(self, message: str, level: NotificationLevel = NotificationLevel.INFO):
        """Envoie un message via Telegram"""
        if not self.enabled:
            return
            
        # Formater le message
        formatted_message = f"{level.value} *TradingLife Bot*\n\n{message}\n\n_{datetime.now().strftime('%H:%M:%S')}_"
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': formatted_message,
            'parse_mode': 'Markdown'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        print(f"Erreur Telegram: {await resp.text()}")
        except Exception as e:
            print(f"Erreur envoi Telegram: {e}")
    
    async def send_trade_notification(self, trade_info: dict):
        """Notification spéciale pour les trades"""
        emoji = "🟢" if trade_info['side'] == 'BUY' else "🔴"
        
        message = f"""
{emoji} *TRADE EXÉCUTÉ*

📈 *Paire:* {trade_info['symbol']}
💰 *Type:* {trade_info['side']}
💵 *Prix:* ${trade_info['price']:.2f}
📊 *Quantité:* {trade_info['amount']}
🎯 *Raison:* {trade_info.get('reason', 'Signal détecté')}

📊 *P&L Total:* ${trade_info.get('total_pnl', 0):.2f}
"""
        await self.send_message(message, NotificationLevel.TRADE)
    
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
        await self.send_message(message, NotificationLevel.TRADE)
    
    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()