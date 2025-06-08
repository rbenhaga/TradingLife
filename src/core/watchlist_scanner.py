"""
Scanner de watchlist pour identifier les meilleures opportunités de trading
"""

import logging
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime, timedelta

class WatchlistScanner:
    """
    Scanner pour identifier les cryptos les plus prometteuses
    basé sur la volatilité, le volume et les opportunités
    """
    
    def __init__(self, exchange_connector, 
                 top_n: int = 10,
                 min_volume_usdt: float = 1_000_000):
        """
        Initialise le scanner
        
        Args:
            exchange_connector: Connecteur à l'exchange
            top_n: Nombre de paires à garder
            min_volume_usdt: Volume minimum en USDT
        """
        self.exchange = exchange_connector
        self.top_n = top_n
        self.min_volume_usdt = min_volume_usdt
        self.logger = logging.getLogger('WatchlistScanner')
        
        self.logger.info(
            f"Scanner de volatilité initialisé - "
            f"Top {top_n} paires, Volume min: {min_volume_usdt/1e6:.1f}M USDT"
        )
    
    def scan_market(self, base_currency: str = 'USDT') -> List[str]:
        """
        Scanne le marché pour trouver les meilleures paires
        
        Args:
            base_currency: Devise de base (USDT par défaut)
            
        Returns:
            Liste des symboles triés par score
        """
        try:
            self.logger.info("🔍 Début du scan de marché...")
            
            # Récupérer toutes les paires avec la devise de base
            markets = self.exchange.exchange.markets
            pairs = [
                symbol for symbol in markets 
                if symbol.endswith(f'/{base_currency}') 
                and markets[symbol]['active']
            ]
            
            self.logger.info(f"Paires {base_currency} disponibles: {pairs[:20]}...")
            
            # Récupérer les tickers pour toutes les paires
            tickers = self.exchange.exchange.fetch_tickers(pairs)
            self.logger.info(f"Nombre de tickers récupérés: {len(tickers)}")
            
            # Analyser et scorer chaque paire
            scored_pairs = []
            
            for symbol, ticker in tickers.items():
                score_data = self._calculate_volatility_score(symbol, ticker)
                if score_data:
                    scored_pairs.append(score_data)
            
            # Trier par score décroissant
            scored_pairs.sort(key=lambda x: x['score'], reverse=True)
            
            # Garder seulement le top N
            top_pairs = scored_pairs[:self.top_n]
            
            # Afficher le résumé
            self.logger.info(f"🏆 Top paires par score de volatilité:")
            for i, pair_data in enumerate(top_pairs[:5]):  # Afficher top 5
                self.logger.info(
                    f"  {pair_data['symbol']}: Score={pair_data['score']:.2f} | "
                    f"Vol={pair_data['volatility']:.2f}% | "
                    f"Volume={pair_data['volume_usdt']/1e6:.1f}M | "
                    f"24h={pair_data['change_24h']:+.2f}%"
                )
            
            # Retourner seulement les symboles
            return [pair['symbol'] for pair in top_pairs]
            
        except Exception as e:
            self.logger.error(f"Erreur lors du scan: {e}")
            # Retourner une liste par défaut en cas d'erreur
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    def _calculate_volatility_score(self, symbol: str, ticker: Dict) -> Dict:
        """
        Calcule le score de volatilité pour une paire
        
        Args:
            symbol: Symbole de la paire
            ticker: Données du ticker
            
        Returns:
            Dict avec les données et le score
        """
        try:
            # Extraire les données du ticker
            volume_usdt = ticker.get('quoteVolume', 0)
            
            # Vérifier le volume minimum
            if volume_usdt < self.min_volume_usdt:
                return None
            
            # Calculer la volatilité (high-low) / average
            high = ticker.get('high', 0)
            low = ticker.get('low', 0)
            last = ticker.get('last', 0)
            
            if low > 0 and last > 0:
                volatility = ((high - low) / last) * 100
            else:
                volatility = 0
            
            # Changement 24h
            change_24h = ticker.get('percentage', 0)
            
            # Calculer le score composite
            # Favoriser: volatilité modérée + volume élevé + momentum
            volatility_score = self._score_volatility(volatility)
            volume_score = self._score_volume(volume_usdt)
            momentum_score = self._score_momentum(change_24h)
            
            # Score final pondéré
            final_score = (
                volatility_score * 0.4 +
                volume_score * 0.3 +
                momentum_score * 0.3
            )
            
            return {
                'symbol': symbol,
                'score': final_score,
                'volatility': volatility,
                'volume_usdt': volume_usdt,
                'change_24h': change_24h,
                'last_price': last,
                'high': high,
                'low': low
            }
            
        except Exception as e:
            self.logger.debug(f"Erreur calcul score pour {symbol}: {e}")
            return None
    
    def _score_volatility(self, volatility: float) -> float:
        """
        Score la volatilité (préférence pour volatilité modérée)
        
        Args:
            volatility: Volatilité en %
            
        Returns:
            Score de 0 à 100
        """
        # Volatilité idéale entre 2% et 10%
        if volatility < 0.5:
            return 0  # Trop stable
        elif volatility < 2:
            return volatility * 25  # Montée linéaire
        elif volatility < 5:
            return 50 + (volatility - 2) * 10  # Optimal
        elif volatility < 10:
            return 80 - (volatility - 5) * 8  # Descente
        else:
            return max(0, 40 - (volatility - 10) * 2)  # Trop volatil
    
    def _score_volume(self, volume_usdt: float) -> float:
        """
        Score le volume (plus c'est élevé, mieux c'est)
        
        Args:
            volume_usdt: Volume en USDT
            
        Returns:
            Score de 0 à 100
        """
        # Score logarithmique pour le volume
        if volume_usdt < self.min_volume_usdt:
            return 0
        elif volume_usdt < 10_000_000:  # 10M
            return 20 + (volume_usdt / 10_000_000) * 30
        elif volume_usdt < 100_000_000:  # 100M
            return 50 + (volume_usdt / 100_000_000) * 30
        else:
            return min(100, 80 + (volume_usdt / 1_000_000_000) * 20)
    
    def _score_momentum(self, change_24h: float) -> float:
        """
        Score le momentum (favorise les mouvements positifs modérés)
        
        Args:
            change_24h: Changement 24h en %
            
        Returns:
            Score de 0 à 100
        """
        # Favorise un momentum positif mais pas excessif
        if change_24h < -10:
            return 10  # Forte baisse
        elif change_24h < 0:
            return 30 + (change_24h + 10) * 2  # Baisse modérée
        elif change_24h < 5:
            return 50 + change_24h * 8  # Hausse modérée (optimal)
        elif change_24h < 20:
            return 90 - (change_24h - 5) * 2  # Hausse forte
        else:
            return max(0, 60 - (change_24h - 20))  # Trop de hausse
    
    def get_pair_analysis(self, symbol: str) -> Dict:
        """
        Analyse approfondie d'une paire spécifique
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Dict avec analyse détaillée
        """
        try:
            # Récupérer les données actuelles
            ticker = self.exchange.exchange.fetch_ticker(symbol)
            
            # Récupérer l'historique pour plus d'analyse
            ohlcv = self.exchange.exchange.fetch_ohlcv(
                symbol, 
                timeframe='1h', 
                limit=24
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Analyser les patterns
            analysis = {
                'symbol': symbol,
                'current_price': ticker['last'],
                'volume_24h': ticker['quoteVolume'],
                'change_24h': ticker['percentage'],
                'volatility': ((ticker['high'] - ticker['low']) / ticker['last']) * 100,
                'spread': ticker['ask'] - ticker['bid'] if ticker['ask'] and ticker['bid'] else 0,
                'hourly_changes': self._calculate_hourly_momentum(df),
                'support_resistance': self._find_support_resistance(df),
                'volume_profile': self._analyze_volume_profile(df)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erreur analyse {symbol}: {e}")
            return {}
    
    def _calculate_hourly_momentum(self, df: pd.DataFrame) -> Dict:
        """Calcule le momentum sur différentes périodes"""
        if len(df) < 24:
            return {}
        
        current = df['close'].iloc[-1]
        
        return {
            '1h': ((current - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100,
            '4h': ((current - df['close'].iloc[-5]) / df['close'].iloc[-5]) * 100,
            '12h': ((current - df['close'].iloc[-13]) / df['close'].iloc[-13]) * 100,
            '24h': ((current - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        }
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Trouve les niveaux de support et résistance"""
        if len(df) < 10:
            return {}
        
        highs = df['high'].values
        lows = df['low'].values
        
        # Support = plus bas récent
        support = np.min(lows[-10:])
        
        # Résistance = plus haut récent
        resistance = np.max(highs[-10:])
        
        # Prix actuel
        current = df['close'].iloc[-1]
        
        return {
            'support': support,
            'resistance': resistance,
            'current': current,
            'position': (current - support) / (resistance - support) if resistance > support else 0.5
        }
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyse le profil de volume"""
        if len(df) < 5:
            return {}
        
        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].mean()
        
        return {
            'recent_vs_average': recent_volume / avg_volume if avg_volume > 0 else 1,
            'trend': 'increasing' if recent_volume > avg_volume * 1.2 else 'decreasing' if recent_volume < avg_volume * 0.8 else 'stable'
        }
    
    def update_filters(self, top_n: int = None, min_volume_usdt: float = None):
        """
        Met à jour les filtres du scanner
        
        Args:
            top_n: Nouveau nombre de paires à garder
            min_volume_usdt: Nouveau volume minimum
        """
        if top_n is not None:
            self.top_n = top_n
        
        if min_volume_usdt is not None:
            self.min_volume_usdt = min_volume_usdt
        
        self.logger.info(
            f"Filtres mis à jour - Top {self.top_n}, "
            f"Volume min: {self.min_volume_usdt/1e6:.1f}M USDT"
        )