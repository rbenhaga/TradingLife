#!/usr/bin/env python3
import asyncio
import os
from dotenv import load_dotenv
from src.notifications.telegram_notifier import TelegramNotifier, NotificationLevel

async def test_telegram():
    load_dotenv()
    
    notifier = TelegramNotifier()
    
    if not notifier.enabled:
        print("❌ Telegram non configuré!")
        return
    
    print("📤 Envoi du message de test...")
    await notifier.send_message(
        "🎉 Bot Telegram configuré avec succès!",
        NotificationLevel.SUCCESS
    )
    
    # Test trade notification
    await notifier.send_trade_notification({
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'price': 65432.10,
        'amount': 0.001,
        'reason': 'RSI oversold + MACD cross',
        'total_pnl': 125.50
    })
    
    print("✅ Messages envoyés!")

if __name__ == "__main__":
    asyncio.run(test_telegram()) 