"""
Module pour le dashboard web
"""

from typing import Dict, List
import dash
from dash import html, dcc
from flask import Flask, render_template, jsonify
from ..core.trading_bot import TradingBot
from ..core.multi_pair_manager import MultiPairManager
from ..utils.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands

class Dashboard:
    """Classe pour le dashboard web"""
    
    def __init__(self):
        """Initialise le dashboard"""
        self.app = dash.Dash(__name__)
        self.setup_layout()
    
    def setup_layout(self):
        """Configure la mise en page du dashboard"""
        self.app.layout = html.Div([
            html.H1("Crypto Trading Bot Dashboard"),
            html.Div(id="status-container"),
            html.Div(id="trades-container")
        ]) 