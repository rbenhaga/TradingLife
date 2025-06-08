"""
Tests unitaires pour les stratégies de trading
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.strategy import Strategy, MultiSignalStrategy
from src.core.weighted_score_engine import WeightedScoreEngine


class TestStrategy(unittest.TestCase):
    """Tests pour la classe Strategy de base"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.symbol = 'BTC/USDT'
        self.timeframe = '15m'
        self.strategy = Strategy(self.symbol, self.timeframe)
    
    def test_initialization(self):
        """Test l'initialisation de la stratégie"""
        self.assertEqual(self.strategy.symbol, self.symbol)
        self.assertEqual(self.strategy.timeframe, self.timeframe)
        self.assertIsNotNone(self.strategy.logger)
    
    def test_should_enter_not_implemented(self):
        """Test que should_enter lève NotImplementedError"""
        df = pd.DataFrame()
        with self.assertRaises(NotImplementedError):
            self.strategy.should_enter(df)
    
    def test_should_exit_not_implemented(self):
        """Test que should_exit lève NotImplementedError"""
        df = pd.DataFrame()
        position = {}
        with self.assertRaises(NotImplementedError):
            self.strategy.should_exit(df, position)


class TestMultiSignalStrategy(unittest.TestCase):
    """Tests pour MultiSignalStrategy"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.symbol = 'BTC/USDT'
        self.strategy = MultiSignalStrategy(self.symbol)
        
        # Créer des données de test
        self.df = self._create_test_dataframe()
    
    def _create_test_dataframe(self):
        """Crée un DataFrame de test avec des données OHLCV"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='15min')
        
        # Générer des prix simulés
        np.random.seed(42)
        close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + np.random.randn(100) * 50,
            'high': close_prices + abs(np.random.randn(100) * 100),
            'low': close_prices - abs(np.random.randn(100) * 100),
            'close': close_prices,
            'volume': 1000000 + np.random.randn(100) * 100000
        })
        
        return df
    
    def test_initialization_with_custom_weights(self):
        """Test l'initialisation avec des poids personnalisés"""
        custom_weights = {
            'rsi': 0.3,
            'bollinger': 0.3,
            'macd': 0.2,
            'volume': 0.2
        }
        
        strategy = MultiSignalStrategy(self.symbol, score_weights=custom_weights)
        self.assertEqual(strategy.rsi_oversold, 30)
        self.assertEqual(strategy.rsi_overbought, 70)
    
    def test_analyze_method_returns_valid_signal(self):
        """Test que analyze retourne un signal valide"""
        signal = self.strategy.analyze(self.df)
        
        self.assertIn('action', signal)
        self.assertIn('score', signal)
        self.assertIn('confidence', signal)
        self.assertIn('details', signal)
        
        # Vérifier les types
        self.assertIn(signal['action'], ['BUY', 'SELL', 'NEUTRAL'])
        self.assertIsInstance(signal['score'], float)
        self.assertIsInstance(signal['confidence'], float)
        self.assertIsInstance(signal['details'], dict)
    
    def test_should_enter_buy_signal(self):
        """Test la détection d'un signal d'achat"""
        # Simuler un score d'achat fort
        with patch.object(self.strategy.score_engine, 'calculate_score') as mock_score:
            mock_score.return_value = {
                'score': 0.6,  # Score d'achat fort
                'confidence': 0.8,
                'details': {}
            }
            
            signal = self.strategy.should_enter(self.df)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal['action'], 'BUY')
            self.assertEqual(signal['type'], 'market')
    
    def test_should_enter_no_signal(self):
        """Test pas de signal d'entrée en zone neutre"""
        # Simuler un score neutre
        with patch.object(self.strategy.score_engine, 'calculate_score') as mock_score:
            mock_score.return_value = {
                'score': 0.1,  # Score neutre
                'confidence': 0.5,
                'details': {}
            }
            
            signal = self.strategy.should_enter(self.df)
            self.assertIsNone(signal)
    
    def test_should_exit_take_profit(self):
        """Test la sortie sur take profit"""
        position = {
            'entry_price': 50000,
            'current_price': 52500,  # +5%
            'quantity': 0.01,
            'side': 'long'
        }
        
        # Simuler un score de vente
        with patch.object(self.strategy.score_engine, 'calculate_score') as mock_score:
            mock_score.return_value = {
                'score': -0.4,  # Score de vente
                'confidence': 0.7,
                'details': {}
            }
            
            signal = self.strategy.should_exit(self.df, position)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal['action'], 'SELL')
            self.assertIn('reason', signal)
    
    def test_should_exit_stop_loss(self):
        """Test la sortie sur stop loss"""
        position = {
            'entry_price': 50000,
            'current_price': 47000,  # -6%
            'quantity': 0.01,
            'side': 'long'
        }
        
        signal = self.strategy.should_exit(self.df, position)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal['action'], 'SELL')
        self.assertIn('stop_loss', signal['reason'].lower())
    
    def test_get_position_size(self):
        """Test le calcul de la taille de position"""
        capital = 10000
        current_price = 50000
        
        size = self.strategy.get_position_size(capital, current_price)
        
        # Vérifier que la taille est dans les limites attendues
        self.assertGreater(size, 0)
        self.assertLessEqual(size * current_price, capital * 0.1)  # Max 10% du capital
    
    def test_analyze_with_insufficient_data(self):
        """Test analyze avec données insuffisantes"""
        small_df = self.df.head(10)  # Pas assez de données pour certains indicateurs
        
        signal = self.strategy.analyze(small_df)
        
        # Devrait retourner un signal neutre
        self.assertEqual(signal['action'], 'NEUTRAL')
        self.assertLess(signal['confidence'], 0.5)


class TestWeightedScoreIntegration(unittest.TestCase):
    """Tests d'intégration avec le score engine"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.strategy = MultiSignalStrategy('ETH/USDT')
        self.df = self._create_trending_dataframe()
    
    def _create_trending_dataframe(self):
        """Crée un DataFrame avec une tendance claire"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='15min')
        
        # Créer une tendance haussière
        trend = np.linspace(3000, 3300, 100)
        noise = np.random.randn(100) * 10
        close_prices = trend + noise
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices - 5,
            'high': close_prices + 10,
            'low': close_prices - 10,
            'close': close_prices,
            'volume': np.linspace(1000000, 1500000, 100) + np.random.randn(100) * 50000
        })
        
        return df
    
    def test_bullish_trend_detection(self):
        """Test la détection d'une tendance haussière"""
        signal = self.strategy.analyze(self.df)
        
        # Dans une tendance haussière claire, on devrait avoir un signal positif
        self.assertGreater(signal['score'], 0)
        self.assertIn(signal['action'], ['BUY', 'NEUTRAL'])
    
    def test_score_details_completeness(self):
        """Test que tous les indicateurs sont présents dans les détails"""
        signal = self.strategy.analyze(self.df)
        
        expected_indicators = ['rsi', 'bollinger', 'macd', 'volume', 'ma_cross', 'momentum', 'volatility']
        
        for indicator in expected_indicators:
            self.assertIn(indicator, signal['details'])
            self.assertIn('signal', signal['details'][indicator])
            self.assertIn('confidence', signal['details'][indicator])


class TestStrategyEdgeCases(unittest.TestCase):
    """Tests des cas limites"""
    
    def test_empty_dataframe(self):
        """Test avec un DataFrame vide"""
        strategy = MultiSignalStrategy('BTC/USDT')
        empty_df = pd.DataFrame()
        
        signal = strategy.analyze(empty_df)
        self.assertEqual(signal['action'], 'NEUTRAL')
        self.assertEqual(signal['confidence'], 0)
    
    def test_extreme_market_conditions(self):
        """Test avec des conditions de marché extrêmes"""
        strategy = MultiSignalStrategy('BTC/USDT')
        
        # Créer des données avec une chute de 50%
        dates = pd.date_range(end=datetime.now(), periods=50, freq='15min')
        close_prices = np.linspace(60000, 30000, 50)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + 100,
            'high': close_prices + 200,
            'low': close_prices - 100,
            'close': close_prices,
            'volume': [1000000] * 50
        })
        
        signal = strategy.analyze(df)
        
        # Dans une chute extrême, on devrait avoir un signal de vente ou neutre
        self.assertIn(signal['action'], ['SELL', 'NEUTRAL'])


if __name__ == '__main__':
    unittest.main()