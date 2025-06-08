"""Binance exchange connector implementation"""

import ccxt.async_support as ccxt
from typing import Dict, List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

class BinanceConnector:
    """Binance exchange connector implementation"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        """
        Initialize Binance connector.
        
        Args:
            api_key (str): API key
            api_secret (str): API secret
            testnet (bool): Use testnet
        """
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'testnet': testnet
            }
        })
        self.testnet = testnet
        logger.info(f"Initialisation du connecteur binance (testnet: {testnet})")

    async def connect(self) -> bool:
        """Test connection to exchange"""
        try:
            await self.exchange.load_markets()
            return True
        except Exception as e:
            logger.error(f"Erreur de connexion à binance: {str(e)}")
            return False

    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker for symbol"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume']
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du ticker {symbol}: {str(e)}")
            return None

    async def get_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 500) -> List[Dict]:
        """Get OHLCV data for symbol"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return [{
                'timestamp': candle[0],
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5]
            } for candle in ohlcv]
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des OHLCV pour {symbol}: {str(e)}")
            return []

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Create a new order"""
        try:
            order = await self.exchange.create_order(symbol, order_type, side, amount, price)
            return {
                'id': order['id'],
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'amount': order['amount'],
                'price': order['price'],
                'status': order['status']
            }
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'ordre pour {symbol}: {str(e)}")
            return None

    async def close(self):
        """Close exchange connection"""
        await self.exchange.close() 