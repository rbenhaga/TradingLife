# src/web/modern_dashboard.py
"""
Dashboard moderne style startup fintech avec visualisations avancées
Utilise React + Chart.js + WebSocket pour temps réel
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List
import json
import asyncio
from datetime import datetime

from ..core.logger import log_info


class ModernDashboard:
    """Dashboard web moderne avec interface futuriste"""
    
    def __init__(self, bot):
        self.bot = bot
        self.app = FastAPI(title="TradingBot Dashboard")
        self.websocket_clients = []
        self._setup_routes()
    
    def _setup_routes(self):
        """Configure les routes de l'API"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self._get_dashboard_html()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_clients.append(websocket)
            
            try:
                while True:
                    # Envoyer les mises à jour
                    data = self._get_realtime_data()
                    await websocket.send_json(data)
                    await asyncio.sleep(1)  # Update chaque seconde
                    
            except WebSocketDisconnect:
                self.websocket_clients.remove(websocket)
    
    def _get_realtime_data(self) -> Dict:
        """Récupère les données temps réel du bot"""
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio': {
                'total_value': self.bot.get_portfolio_value(),
                'positions': self.bot.get_open_positions(),
                'daily_pnl': self.bot.get_daily_pnl(),
                'total_pnl': self.bot.get_total_pnl()
            },
            'performance': {
                'sharpe_ratio': self.bot.get_sharpe_ratio(),
                'win_rate': self.bot.get_win_rate(),
                'profit_factor': self.bot.get_profit_factor(),
                'max_drawdown': self.bot.get_max_drawdown()
            },
            'market_analysis': self.bot.get_market_analysis(),
            'recent_trades': self.bot.get_recent_trades(10),
            'active_signals': self.bot.get_active_signals()
        }
    
    def _get_dashboard_html(self) -> str:
        """HTML du dashboard moderne"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradingBot - Advanced Dashboard</title>
    
    <!-- Styles -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    
    <style>
        :root {
            --primary: #0ea5e9;
            --secondary: #8b5cf6;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --dark: #0f172a;
            --darker: #020617;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--darker);
            color: #e2e8f0;
        }
        
        .glass-card {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .glow-primary {
            box-shadow: 0 0 30px rgba(14, 165, 233, 0.3);
        }
        
        .glow-success {
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
        }
        
        .glow-danger {
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        }
        
        .gradient-text {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
        }
        
        /* Animations */
        @keyframes pulse-glow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .pulse-glow {
            animation: pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--dark);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 4px;
        }
    </style>
</head>
<body class="min-h-screen">
    <div id="root"></div>
    
    <script type="text/babel">
        const { useState, useEffect, useRef } = React;
        
        // Composant principal
        function TradingDashboard() {
            const [data, setData] = useState(null);
            const [connected, setConnected] = useState(false);
            const ws = useRef(null);
            const chartRefs = useRef({});
            
            // WebSocket connection
            useEffect(() => {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws.current = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.current.onopen = () => {
                    setConnected(true);
                    console.log('WebSocket connected');
                };
                
                ws.current.onmessage = (event) => {
                    const newData = JSON.parse(event.data);
                    setData(newData);
                    updateCharts(newData);
                };
                
                ws.current.onclose = () => {
                    setConnected(false);
                    console.log('WebSocket disconnected');
                };
                
                return () => {
                    ws.current.close();
                };
            }, []);
            
            // Fonction pour formatter les nombres
            const formatNumber = (num, decimals = 2) => {
                return new Intl.NumberFormat('en-US', {
                    minimumFractionDigits: decimals,
                    maximumFractionDigits: decimals
                }).format(num);
            };
            
            // Fonction pour formatter les pourcentages
            const formatPercent = (num) => {
                const formatted = formatNumber(num, 2);
                return num >= 0 ? `+${formatted}%` : `${formatted}%`;
            };
            
            // Mise à jour des graphiques
            const updateCharts = (newData) => {
                // Mise à jour du graphique de performance
                if (chartRefs.current.performance && newData.portfolio) {
                    // Logic pour mettre à jour le graphique
                }
            };
            
            if (!data) {
                return (
                    <div className="min-h-screen flex items-center justify-center">
                        <div className="text-center">
                            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mx-auto mb-4"></div>
                            <p className="text-gray-400">Connecting to trading bot...</p>
                        </div>
                    </div>
                );
            }
            
            return (
                <div className="min-h-screen p-6">
                    {/* Header */}
                    <header className="mb-8">
                        <div className="flex items-center justify-between">
                            <div>
                                <h1 className="text-4xl font-bold gradient-text mb-2">
                                    Trading Bot Dashboard
                                </h1>
                                <p className="text-gray-400">
                                    Real-time trading performance & analytics
                                </p>
                            </div>
                            <div className="flex items-center space-x-4">
                                <div className={`flex items-center ${connected ? 'text-success' : 'text-danger'}`}>
                                    <div className={`w-3 h-3 rounded-full ${connected ? 'bg-success' : 'bg-danger'} mr-2 pulse-glow`}></div>
                                    {connected ? 'Connected' : 'Disconnected'}
                                </div>
                            </div>
                        </div>
                    </header>
                    
                    {/* KPI Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                        <KPICard
                            title="Portfolio Value"
                            value={`$${formatNumber(data.portfolio.total_value)}`}
                            change={data.portfolio.daily_pnl}
                            trend={data.portfolio.daily_pnl >= 0 ? 'up' : 'down'}
                            icon="💰"
                        />
                        <KPICard
                            title="Total P&L"
                            value={`$${formatNumber(data.portfolio.total_pnl)}`}
                            change={(data.portfolio.total_pnl / 10000) * 100}
                            trend={data.portfolio.total_pnl >= 0 ? 'up' : 'down'}
                            icon="📈"
                        />
                        <KPICard
                            title="Win Rate"
                            value={`${formatNumber(data.performance.win_rate)}%`}
                            subtitle={`Sharpe: ${formatNumber(data.performance.sharpe_ratio, 2)}`}
                            icon="🎯"
                        />
                        <KPICard
                            title="Active Positions"
                            value={data.portfolio.positions.length}
                            subtitle={`Signals: ${data.active_signals.length}`}
                            icon="📊"
                        />
                    </div>
                    
                    {/* Main Content Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Chart Section */}
                        <div className="lg:col-span-2 space-y-6">
                            {/* Performance Chart */}
                            <div className="glass-card p-6">
                                <h2 className="text-xl font-semibold mb-4">Portfolio Performance</h2>
                                <div className="chart-container">
                                    <canvas ref={el => chartRefs.current.performance = el}></canvas>
                                </div>
                            </div>
                            
                            {/* Positions Table */}
                            <div className="glass-card p-6">
                                <h2 className="text-xl font-semibold mb-4">Open Positions</h2>
                                <PositionsTable positions={data.portfolio.positions} />
                            </div>
                        </div>
                        
                        {/* Sidebar */}
                        <div className="space-y-6">
                            {/* Market Analysis */}
                            <div className="glass-card p-6">
                                <h2 className="text-xl font-semibold mb-4">Market Analysis</h2>
                                <MarketAnalysis analysis={data.market_analysis} />
                            </div>
                            
                            {/* Recent Trades */}
                            <div className="glass-card p-6">
                                <h2 className="text-xl font-semibold mb-4">Recent Trades</h2>
                                <RecentTrades trades={data.recent_trades} />
                            </div>
                            
                            {/* Active Signals */}
                            <div className="glass-card p-6">
                                <h2 className="text-xl font-semibold mb-4">Active Signals</h2>
                                <ActiveSignals signals={data.active_signals} />
                            </div>
                        </div>
                    </div>
                </div>
            );
        }
        
        // Composant KPI Card
        function KPICard({ title, value, change, trend, subtitle, icon }) {
            const isPositive = trend === 'up';
            
            return (
                <div className={`glass-card p-6 ${isPositive ? 'glow-success' : 'glow-danger'}`}>
                    <div className="flex items-start justify-between">
                        <div className="flex-1">
                            <p className="text-gray-400 text-sm mb-1">{title}</p>
                            <p className="text-3xl font-bold mb-2">{value}</p>
                            {change !== undefined && (
                                <p className={`text-sm font-medium ${isPositive ? 'text-success' : 'text-danger'}`}>
                                    {formatPercent(change)}
                                </p>
                            )}
                            {subtitle && (
                                <p className="text-gray-400 text-sm mt-1">{subtitle}</p>
                            )}
                        </div>
                        <div className="text-4xl opacity-50">{icon}</div>
                    </div>
                </div>
            );
        }
        
        // Composant Positions Table
        function PositionsTable({ positions }) {
            return (
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="text-left text-gray-400 text-sm">
                                <th className="pb-3">Symbol</th>
                                <th className="pb-3">Side</th>
                                <th className="pb-3 text-right">Entry</th>
                                <th className="pb-3 text-right">Current</th>
                                <th className="pb-3 text-right">P&L</th>
                                <th className="pb-3 text-right">Size</th>
                            </tr>
                        </thead>
                        <tbody>
                            {positions.map((pos, idx) => (
                                <tr key={idx} className="border-t border-gray-800">
                                    <td className="py-3 font-medium">{pos.symbol}</td>
                                    <td className="py-3">
                                        <span className={`px-2 py-1 rounded text-xs ${
                                            pos.side === 'LONG' ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'
                                        }`}>
                                            {pos.side}
                                        </span>
                                    </td>
                                    <td className="py-3 text-right">${formatNumber(pos.entry_price)}</td>
                                    <td className="py-3 text-right">${formatNumber(pos.current_price)}</td>
                                    <td className={`py-3 text-right font-medium ${
                                        pos.pnl >= 0 ? 'text-success' : 'text-danger'
                                    }`}>
                                        ${formatNumber(pos.pnl)}
                                    </td>
                                    <td className="py-3 text-right">{formatNumber(pos.size, 4)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            );
        }
        
        // Composant Market Analysis
        function MarketAnalysis({ analysis }) {
            return (
                <div className="space-y-4">
                    {Object.entries(analysis).map(([symbol, data]) => (
                        <div key={symbol} className="border-b border-gray-800 pb-3">
                            <div className="flex justify-between items-center mb-2">
                                <h3 className="font-medium">{symbol}</h3>
                                <span className={`text-sm px-2 py-1 rounded ${
                                    data.regime === 'TRENDING_UP' ? 'bg-success/20 text-success' :
                                    data.regime === 'TRENDING_DOWN' ? 'bg-danger/20 text-danger' :
                                    'bg-warning/20 text-warning'
                                }`}>
                                    {data.regime}
                                </span>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-sm">
                                <div>
                                    <span className="text-gray-400">Score:</span>
                                    <span className="ml-2 font-medium">{formatNumber(data.score, 3)}</span>
                                </div>
                                <div>
                                    <span className="text-gray-400">Vol:</span>
                                    <span className="ml-2 font-medium">{formatNumber(data.volatility, 2)}%</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            );
        }
        
        // Composant Recent Trades
        function RecentTrades({ trades }) {
            return (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                    {trades.map((trade, idx) => (
                        <div key={idx} className="flex items-center justify-between p-2 rounded hover:bg-gray-800/50">
                            <div className="flex items-center space-x-3">
                                <div className={`w-2 h-2 rounded-full ${
                                    trade.side === 'BUY' ? 'bg-success' : 'bg-danger'
                                }`}></div>
                                <div>
                                    <p className="text-sm font-medium">{trade.symbol}</p>
                                    <p className="text-xs text-gray-400">{new Date(trade.timestamp).toLocaleTimeString()}</p>
                                </div>
                            </div>
                            <div className="text-right">
                                <p className={`text-sm font-medium ${
                                    trade.pnl >= 0 ? 'text-success' : 'text-danger'
                                }`}>
                                    ${formatNumber(trade.pnl)}
                                </p>
                                <p className="text-xs text-gray-400">{trade.side}</p>
                            </div>
                        </div>
                    ))}
                </div>
            );
        }
        
        // Composant Active Signals
        function ActiveSignals({ signals }) {
            return (
                <div className="space-y-3">
                    {signals.map((signal, idx) => (
                        <div key={idx} className="p-3 rounded border border-gray-700">
                            <div className="flex justify-between items-start mb-2">
                                <h4 className="font-medium">{signal.symbol}</h4>
                                <span className={`text-xs px-2 py-1 rounded ${
                                    signal.action === 'BUY' ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'
                                }`}>
                                    {signal.action}
                                </span>
                            </div>
                            <div className="text-sm text-gray-400">
                                <p>Confidence: {formatNumber(signal.confidence * 100)}%</p>
                                <p>Score: {formatNumber(signal.score, 3)}</p>
                                <p className="text-xs mt-1">{signal.reason}</p>
                            </div>
                        </div>
                    ))}
                </div>
            );
        }
        
        // Fonction de formatage des pourcentages
        function formatPercent(value) {
            const formatted = formatNumber(value, 2);
            return value >= 0 ? `+${formatted}%` : `${formatted}%`;
        }
        
        // Fonction de formatage des nombres
        function formatNumber(num, decimals = 2) {
            if (num === null || num === undefined) return '0';
            return parseFloat(num).toFixed(decimals);
        }
        
        // Render
        ReactDOM.render(<TradingDashboard />, document.getElementById('root'));
    </script>
</body>
</html>
"""

    async def run(self, host: str = "127.0.0.0", port: int = 8000):
        """
        Lance le serveur FastAPI pour le dashboard.
        """
        import uvicorn
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    async def start_update_loop(self, interval: float = 1.0):
        """
        Boucle d'envoi périodique des données à tous les clients WebSocket connectés.
        """
        while True:
            if self.websocket_clients:
                data = self._get_realtime_data()
                for ws in self.websocket_clients[:]:
                    try:
                        await ws.send_json(data)
                    except Exception:
                        # Si le client est déconnecté, on le retire
                        self.websocket_clients.remove(ws)
            await asyncio.sleep(interval)