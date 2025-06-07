"""
Module de sélection dynamique des meilleures paires à trader
Scanne en continu les paires les plus volatiles et liquides
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import ccxt
import aiohttp
import json
from src.core.logger import log_info, log_debug, log_warning

class DynamicWatchlist:
    """Scanner de volatilité pour identifier les meilleures opportunités"""
    
    def __init__(self, exchange: ccxt.Exchange, 
                 min_volume_usdt: float = 1_000_000,  # 1M USDT minimum
                 top_n: int = 10,
                 update_interval: int = 3600):  # 1 heure
        """
        Initialise le scanner
        
        Args:
            exchange: Instance CCXT configurée
            min_volume_usdt: Volume minimum en USDT sur 24h
            top_n: Nombre de paires à garder dans la watchlist
            update_interval: Intervalle de mise à jour en secondes
        """
        self.exchange = exchange
        self.min_volume_usdt = min_volume_usdt
        self.top_n = top_n
        self.update_interval = update_interval
        self.watchlist = []
        self.scores = {}
        
        # Liste des paires disponibles sur le testnet Binance
        self.available_pairs = [
            'BTC/USDT', 'BTC/BUSD', 'BNB/USDT', 'BNB/BUSD', 'BNB/BTC',
            'ETH/USDT', 'ETH/BUSD', 'ETH/BTC', 'LTC/USDT', 'LTC/BUSD',
            'LTC/BTC', 'LTC/BNB', 'TRX/USDT', 'TRX/BUSD', 'TRX/BTC',
            'TRX/BNB', 'XRP/USDT', 'XRP/BUSD', 'XRP/BTC', 'XRP/BNB'
        ]
        
        # URL de l'API testnet Binance
        self.base_url = "https://testnet.binance.vision/api/v3"
        
        log_info(f"Scanner de volatilité initialisé - Top {top_n} paires, Volume min: {min_volume_usdt/1e6:.1f}M USDT")
        log_info(f"Paires disponibles sur le testnet: {len(self.available_pairs)}")
    
    async def fetch_ticker(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        """
        Récupère le ticker pour une paire donnée
        
        Args:
            session: Session aiohttp
            symbol: Symbole de la paire
            
        Returns:
            Dictionnaire contenant les données du ticker
        """
        try:
            # Convertir le format de la paire (BTC/USDT -> BTCUSDT)
            formatted_symbol = symbol.replace('/', '')
            
            # Construire l'URL
            url = f"{self.base_url}/ticker/24hr?symbol={formatted_symbol}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'symbol': symbol,
                        'last': float(data['lastPrice']),
                        'high': float(data['highPrice']),
                        'low': float(data['lowPrice']),
                        'volume': float(data['volume']),
                        'quoteVolume': float(data['quoteVolume']),
                        'percentage': float(data['priceChangePercent']),
                        'bid': float(data['bidPrice']),
                        'ask': float(data['askPrice'])
                    }
                else:
                    log_warning(f"Erreur HTTP {response.status} pour {symbol}")
                    return None
        except Exception as e:
            log_warning(f"Erreur lors de la récupération du ticker pour {symbol}: {str(e)}")
            return None
    
    async def scan_market(self) -> List[str]:
        """
        Scanne le marché et retourne les meilleures paires
        
        Returns:
            Liste des symboles sélectionnés
        """
        try:
            log_info("🔍 Début du scan de marché...")
            
            # Utiliser uniquement les paires disponibles sur le testnet
            usdt_pairs = [
                symbol for symbol in self.available_pairs
                if symbol.endswith('/USDT')  # On ne garde que les paires USDT pour commencer
            ]
            
            log_info(f"Paires USDT disponibles: {usdt_pairs}")
            
            # Récupérer les tickers en parallèle
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetch_ticker(session, symbol) for symbol in usdt_pairs]
                tickers_list = await asyncio.gather(*tasks)
                
                # Filtrer les tickers None
                tickers = {
                    symbol: ticker for symbol, ticker in zip(usdt_pairs, tickers_list)
                    if ticker is not None
                }
            
            log_info(f"Nombre de tickers récupérés: {len(tickers)}")
            
            if not tickers:
                log_warning("Aucun ticker récupéré")
                return self.watchlist
                
            # Afficher un exemple de ticker pour debug
            first_symbol = next(iter(tickers))
            log_info(f"Exemple de ticker pour {first_symbol}: {tickers[first_symbol]}")
            
            # Calculer les scores
            scores_data = []
            
            for symbol, ticker in tickers.items():
                try:
                    if not ticker:
                        log_debug(f"Ticker manquant pour {symbol}")
                        continue
                    
                    # En mode testnet, on ignore le volume minimum
                    volume_24h = ticker.get('quoteVolume', 0)
                    
                    # Calculer les métriques
                    price_change_pct = ticker.get('percentage', 0)
                    high = ticker.get('high', 0)
                    low = ticker.get('low', 0)
                    last = ticker.get('last', 0)
                    
                    log_debug(f"Données pour {symbol}: volume={volume_24h}, change={price_change_pct}%, high={high}, low={low}, last={last}")
                    
                    if high > 0 and low > 0 and last > 0:
                        # Volatilité = (high - low) / last * 100
                        volatility = ((high - low) / last) * 100
                        
                        # Momentum = changement sur 24h
                        momentum = abs(price_change_pct)
                        
                        # Volume score (log scale pour normaliser)
                        volume_score = np.log10(max(volume_24h, 1) / 1e6)  # En millions, minimum 1
                        
                        # Score composite
                        # Pondération : 40% volatilité, 30% volume, 30% momentum
                        score = (volatility * 0.4) + (volume_score * 3.0) + (momentum * 0.3)
                        
                        scores_data.append({
                            'symbol': symbol,
                            'volume_24h': volume_24h,
                            'volatility': volatility,
                            'momentum': momentum,
                            'volume_score': volume_score,
                            'score': score,
                            'price': last,
                            'change_24h': price_change_pct
                        })
                except Exception as e:
                    log_warning(f"Erreur lors du traitement du ticker {symbol}: {str(e)}")
                    continue
            
            # Créer DataFrame et trier par score
            df = pd.DataFrame(scores_data)
            
            if len(df) == 0:
                log_warning("Aucune paire ne correspond aux critères. Vérifiez les tickers.")
                return self.watchlist
            
            # Trier par score décroissant
            df = df.sort_values('score', ascending=False)
            
            # Prendre le top N
            top_pairs = df.head(self.top_n)
            
            # Mettre à jour la watchlist
            self.watchlist = top_pairs['symbol'].tolist()
            self.scores = df.set_index('symbol').to_dict('index')
            
            # Logger les résultats
            log_info("🏆 Top paires par score de volatilité:")
            for _, row in top_pairs.iterrows():
                log_info(
                    f"  {row['symbol']}: Score={row['score']:.2f} | "
                    f"Vol={row['volatility']:.2f}% | "
                    f"Volume={row['volume_24h']/1e6:.1f}M | "
                    f"24h={row['change_24h']:+.2f}%"
                )
            
            return self.watchlist
            
        except Exception as e:
            log_warning(f"Erreur lors du scan: {str(e)}")
            import traceback
            log_warning(f"Traceback: {traceback.format_exc()}")
            return self.watchlist
    
    def get_pair_metrics(self, symbol: str) -> Dict:
        """
        Retourne les métriques d'une paire
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Dictionnaire des métriques
        """
        return self.scores.get(symbol, {})
    
    def should_trade_pair(self, symbol: str) -> bool:
        """
        Détermine si une paire devrait être tradée
        
        Args:
            symbol: Symbole à vérifier
            
        Returns:
            True si la paire est dans la watchlist
        """
        return symbol in self.watchlist
    
    async def run_scanner_loop(self):
        """Boucle principale du scanner"""
        while True:
            try:
                await self.scan_market()
                
                # Attendre avant le prochain scan
                log_debug(f"Prochain scan dans {self.update_interval/60:.0f} minutes")
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                log_warning(f"Erreur dans la boucle du scanner: {str(e)}")
                await asyncio.sleep(60)  # Attendre 1 minute en cas d'erreur
    
    def get_trading_params(self, symbol: str) -> Dict:
        """
        Retourne les paramètres de trading optimaux pour une paire
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Dictionnaire des paramètres
        """
        metrics = self.get_pair_metrics(symbol)
        
        if not metrics:
            # Paramètres par défaut
            return {
                'position_size': 0.02,  # 2% du capital
                'stop_loss': 0.02,      # 2%
                'take_profit': 0.05,    # 5%
                'timeframe': '5m'
            }
        
        # Adapter les paramètres selon la volatilité
        volatility = metrics.get('volatility', 2.0)
        
        if volatility > 5.0:  # Très volatil
            return {
                'position_size': 0.01,  # 1% seulement
                'stop_loss': 0.03,      # 3%
                'take_profit': 0.08,    # 8%
                'timeframe': '1m'       # Timeframe court
            }
        elif volatility > 3.0:  # Volatilité normale
            return {
                'position_size': 0.02,  # 2%
                'stop_loss': 0.02,      # 2%
                'take_profit': 0.05,    # 5%
                'timeframe': '5m'
            }
        else:  # Peu volatil
            return {
                'position_size': 0.03,  # 3%
                'stop_loss': 0.015,     # 1.5%
                'take_profit': 0.03,    # 3%
                'timeframe': '15m'      # Timeframe plus long
            }
    
    def get_priority_order(self) -> List[Tuple[str, float]]:
        """
        Retourne la liste des paires ordonnées par priorité
        
        Returns:
            Liste de tuples (symbol, score)
        """
        return [
            (symbol, self.scores[symbol]['score']) 
            for symbol in self.watchlist
            if symbol in self.scores
        ]