"""
Dashboard Web pour le Trading Bot
Interface de monitoring temps réel avec FastAPI et WebSocket
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import asyncio
import json
from datetime import datetime
from pathlib import Path
import secrets
from fastapi import WebSocketDisconnect, ConnectionClosed

from ..core.logger import log_info, log_error

# Configuration de sécurité
security = HTTPBasic()
DASHBOARD_USERNAME = "admin"
DASHBOARD_PASSWORD = "tradingbot123"  # À changer en production !


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Vérifie les credentials pour l'accès au dashboard"""
    correct_username = secrets.compare_digest(credentials.username, DASHBOARD_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, DASHBOARD_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username


class Dashboard:
    """Classe principale du dashboard"""
    
    def __init__(self, trading_bot=None):
        """
        Initialise le dashboard
        
        Args:
            trading_bot: Instance du TradingBot à monitorer
        """
        self.app = FastAPI(title="Crypto Trading Bot Dashboard")
        self.bot = trading_bot
        self.websocket_clients: List[WebSocket] = []
        
        # Configuration CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Configuration des routes
        self._setup_routes()
        
        # Créer les répertoires statiques
        self.static_dir = Path(__file__).parent / "static"
        self.static_dir.mkdir(exist_ok=True)
        
        log_info("Dashboard initialisé")
    
    def _setup_routes(self):
        """Configure toutes les routes de l'API"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root(username: str = Depends(verify_credentials)):
            """Page principale du dashboard"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/status")
        async def get_status(username: str = Depends(verify_credentials)):
            """Retourne le status actuel du bot"""
            if not self.bot:
                raise HTTPException(status_code=503, detail="Bot non disponible")
            
            return self.bot.get_status()
        
        @self.app.get("/api/trades")
        async def get_trades(
            limit: int = 50,
            username: str = Depends(verify_credentials)
        ):
            """Retourne l'historique des trades"""
            if not self.bot:
                raise HTTPException(status_code=503, detail="Bot non disponible")
            
            # Récupérer les trades depuis le pair_manager
            trades = []
            if self.bot.pair_manager:
                for symbol, perf in self.bot.pair_manager.performance.items():
                    trades.append({
                        'symbol': symbol,
                        'trades': perf['trades'],
                        'wins': perf['wins'],
                        'losses': perf['losses'],
                        'pnl': perf['pnl'],
                        'last_trade': perf['last_trade'].isoformat() if perf['last_trade'] else None
                    })
            
            return trades[:limit]
        
        @self.app.get("/api/positions")
        async def get_positions(username: str = Depends(verify_credentials)):
            """Retourne les positions ouvertes"""
            if not self.bot or not self.bot.pair_manager:
                return []
            
            positions = []
            for symbol, pos in self.bot.pair_manager.positions.items():
                # Calculer le PnL non réalisé
                current_price = self.bot.websocket_feed.get_ticker(symbol)
                if current_price:
                    entry_value = pos['entry_price'] * pos['size']
                    current_value = current_price['last'] * pos['size']
                    unrealized_pnl = current_value - entry_value
                    unrealized_pnl_pct = ((current_value / entry_value) - 1) * 100
                else:
                    unrealized_pnl = 0
                    unrealized_pnl_pct = 0
                
                positions.append({
                    'symbol': symbol,
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'current_price': current_price['last'] if current_price else pos['entry_price'],
                    'size': pos['size'],
                    'entry_time': pos['entry_time'].isoformat(),
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'stop_loss': pos.get('stop_loss'),
                    'take_profit': pos.get('take_profit')
                })
            
            return positions
        
        @self.app.get("/api/performance")
        async def get_performance(username: str = Depends(verify_credentials)):
            """Retourne les métriques de performance"""
            if not self.bot:
                raise HTTPException(status_code=503, detail="Bot non disponible")
            
            if self.bot.pair_manager:
                perf = self.bot.pair_manager.get_performance_summary()
            else:
                perf = {}
            
            # Ajouter les métriques de risque
            if self.bot.risk_manager:
                risk_metrics = self.bot.risk_manager.get_risk_metrics(
                    self.bot.config['trading']['initial_balance']
                )
                perf['risk'] = {
                    'current_drawdown': risk_metrics.current_drawdown,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'risk_score': risk_metrics.risk_score,
                    'sharpe_ratio': risk_metrics.sharpe_ratio
                }
            
            return perf
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket pour les mises à jour temps réel"""
            try:
                await websocket.accept()
                self.websocket_clients.append(websocket)
                log_info(f"Nouvelle connexion WebSocket établie. Total clients: {len(self.websocket_clients)}")
                
                try:
                    # Envoyer le status initial
                    if self.bot:
                        await websocket.send_json({
                            'type': 'status',
                            'data': self.bot.get_status()
                        })
                    
                    # Garder la connexion ouverte
                    while True:
                        try:
                            # Attendre des messages du client (ping/pong)
                            data = await websocket.receive_text()
                            
                            if data == "ping":
                                await websocket.send_text("pong")
                                
                        except Exception as e:
                            if isinstance(e, (WebSocketDisconnect, ConnectionClosed)):
                                log_info("Client WebSocket déconnecté")
                                break
                            else:
                                log_error(f"Erreur WebSocket: {str(e)}")
                                break
                        
                except Exception as e:
                    log_error(f"Erreur dans la boucle WebSocket: {str(e)}")
                
            except Exception as e:
                log_error(f"Erreur lors de l'acceptation WebSocket: {str(e)}")
            
            finally:
                if websocket in self.websocket_clients:
                    self.websocket_clients.remove(websocket)
                    log_info(f"Client WebSocket retiré. Total clients: {len(self.websocket_clients)}")
        
        @self.app.post("/api/control/pause")
        async def pause_trading(username: str = Depends(verify_credentials)):
            """Met en pause le trading"""
            if not self.bot:
                raise HTTPException(status_code=503, detail="Bot non disponible")
            
            await self.bot._pause_trading()
            return {"status": "paused"}
        
        @self.app.post("/api/control/resume")
        async def resume_trading(username: str = Depends(verify_credentials)):
            """Reprend le trading"""
            if not self.bot:
                raise HTTPException(status_code=503, detail="Bot non disponible")
            
            if self.bot.status.state == BotState.PAUSED:
                self.bot.status.state = BotState.RUNNING
                return {"status": "resumed"}
            
            return {"status": "already_running"}
    
    async def broadcast_update(self, update_type: str, data: Dict):
        """Diffuse une mise à jour à tous les clients WebSocket"""
        if not self.websocket_clients:
            return
            
        message = {
            'type': update_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Envoyer à tous les clients connectés
        disconnected = []
        for websocket in self.websocket_clients:
            try:
                await websocket.send_json(message)
            except Exception as e:
                if isinstance(e, (WebSocketDisconnect, ConnectionClosed)):
                    log_info("Client WebSocket déconnecté lors du broadcast")
                else:
                    log_error(f"Erreur lors du broadcast: {str(e)}")
                disconnected.append(websocket)
        
        # Nettoyer les connexions fermées
        for ws in disconnected:
            if ws in self.websocket_clients:
                self.websocket_clients.remove(ws)
                log_info(f"Client WebSocket retiré après erreur. Total clients: {len(self.websocket_clients)}")
    
    def _get_dashboard_html(self) -> str:
        """Retourne le HTML de la page principale"""
        return """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Trading Bot Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .metric-card {
            @apply bg-white rounded-lg shadow-md p-4 m-2;
        }
        .status-running { @apply text-green-500; }
        .status-paused { @apply text-yellow-500; }
        .status-error { @apply text-red-500; }
        .profit { @apply text-green-600 font-bold; }
        .loss { @apply text-red-600 font-bold; }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-gray-800">🤖 Crypto Trading Bot Dashboard</h1>
            <div id="status" class="mt-2 text-lg"></div>
            <div id="connection-status" class="mt-2 text-sm text-gray-600"></div>
        </header>
        
        <!-- Métriques principales -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="metric-card">
                <h3 class="text-gray-600 text-sm">État</h3>
                <p id="bot-state" class="text-2xl font-bold">-</p>
            </div>
            <div class="metric-card">
                <h3 class="text-gray-600 text-sm">P&L Total</h3>
                <p id="total-pnl" class="text-2xl font-bold">-</p>
            </div>
            <div class="metric-card">
                <h3 class="text-gray-600 text-sm">Positions Ouvertes</h3>
                <p id="open-positions" class="text-2xl font-bold">-</p>
            </div>
            <div class="metric-card">
                <h3 class="text-gray-600 text-sm">Win Rate</h3>
                <p id="win-rate" class="text-2xl font-bold">-</p>
            </div>
        </div>
        
        <!-- Graphique d'équité -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-bold mb-4">Courbe d'Équité</h2>
            <canvas id="equityChart" height="100"></canvas>
        </div>
        
        <!-- Positions ouvertes -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-bold mb-4">Positions Ouvertes</h2>
            <div id="positions-table" class="overflow-x-auto"></div>
        </div>
        
        <!-- Historique des trades -->
        <div class="bg-white rounded-lg shadow-md p-6">
            <h2 class="text-xl font-bold mb-4">Historique des Trades</h2>
            <div id="trades-table" class="overflow-x-auto"></div>
        </div>
        
        <!-- Contrôles -->
        <div class="fixed bottom-4 right-4">
            <button id="pause-btn" class="bg-yellow-500 text-white px-4 py-2 rounded mr-2 hover:bg-yellow-600">
                ⏸️ Pause
            </button>
            <button id="resume-btn" class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 hidden">
                ▶️ Reprendre
            </button>
        </div>
    </div>
    
    <script>
        // Configuration WebSocket
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        const reconnectDelay = 5000; // 5 secondes
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connecté');
                document.getElementById('connection-status').textContent = 'Connecté';
                document.getElementById('connection-status').className = 'mt-2 text-sm text-green-600';
                reconnectAttempts = 0;
            };
            
            ws.onclose = () => {
                console.log('WebSocket déconnecté');
                document.getElementById('connection-status').textContent = 'Déconnecté - Tentative de reconnexion...';
                document.getElementById('connection-status').className = 'mt-2 text-sm text-red-600';
                
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    setTimeout(connectWebSocket, reconnectDelay);
                } else {
                    document.getElementById('connection-status').textContent = 'Déconnecté - Veuillez rafraîchir la page';
                }
            };
            
            ws.onerror = (error) => {
                console.error('Erreur WebSocket:', error);
                document.getElementById('connection-status').textContent = 'Erreur de connexion';
                document.getElementById('connection-status').className = 'mt-2 text-sm text-red-600';
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'status') {
                    updateStatus(message.data);
                } else if (message.type === 'trade') {
                    updateDashboard();
                }
            };
        }
        
        // Chart.js pour la courbe d'équité
        const ctx = document.getElementById('equityChart').getContext('2d');
        const equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Capital',
                    data: [],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
        
        // Mise à jour des données
        async function updateDashboard() {
            try {
                // Status
                const statusResp = await fetch('/api/status');
                if (!statusResp.ok) {
                    throw new Error(`Erreur HTTP: ${statusResp.status}`);
                }
                const status = await statusResp.json();
                updateStatus(status);
                
                // Performance
                const perfResp = await fetch('/api/performance');
                if (!perfResp.ok) {
                    throw new Error(`Erreur HTTP: ${perfResp.status}`);
                }
                const performance = await perfResp.json();
                updatePerformance(performance);
                
                // Positions
                const posResp = await fetch('/api/positions');
                if (!posResp.ok) {
                    throw new Error(`Erreur HTTP: ${posResp.status}`);
                }
                const positions = await posResp.json();
                updatePositions(positions);
                
                // Trades
                const tradesResp = await fetch('/api/trades');
                if (!tradesResp.ok) {
                    throw new Error(`Erreur HTTP: ${tradesResp.status}`);
                }
                const trades = await tradesResp.json();
                updateTrades(trades);
                
            } catch (error) {
                console.error('Erreur mise à jour:', error);
                document.getElementById('connection-status').textContent = `Erreur: ${error.message}`;
                document.getElementById('connection-status').className = 'mt-2 text-sm text-red-600';
            }
        }
        
        function updateStatus(status) {
            // État du bot
            const stateEl = document.getElementById('bot-state');
            stateEl.textContent = status.state.toUpperCase();
            stateEl.className = `text-2xl font-bold status-${status.state}`;
            
            // Uptime
            document.getElementById('status').textContent = 
                `Uptime: ${status.uptime} | Dernière MAJ: ${new Date(status.last_update).toLocaleTimeString()}`;
        }
        
        function updatePerformance(perf) {
            // P&L Total
            const pnlEl = document.getElementById('total-pnl');
            const pnl = perf.total_pnl || 0;
            pnlEl.textContent = `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)} USDT`;
            pnlEl.className = `text-2xl font-bold ${pnl >= 0 ? 'profit' : 'loss'}`;
            
            // Positions ouvertes
            document.getElementById('open-positions').textContent = perf.positions_open || 0;
            
            // Win Rate
            const winRate = perf.win_rate || 0;
            document.getElementById('win-rate').textContent = `${winRate.toFixed(1)}%`;
            
            // Mise à jour du graphique
            if (perf.equity_curve) {
                equityChart.data.labels = perf.equity_curve.map((_, i) => i);
                equityChart.data.datasets[0].data = perf.equity_curve;
                equityChart.update();
            }
        }
        
        function updatePositions(positions) {
            const container = document.getElementById('positions-table');
            
            if (positions.length === 0) {
                container.innerHTML = '<p class="text-gray-500">Aucune position ouverte</p>';
                return;
            }
            
            let html = `
                <table class="min-w-full">
                    <thead>
                        <tr class="border-b">
                            <th class="text-left py-2">Paire</th>
                            <th class="text-left py-2">Côté</th>
                            <th class="text-right py-2">Entrée</th>
                            <th class="text-right py-2">Actuel</th>
                            <th class="text-right py-2">Taille</th>
                            <th class="text-right py-2">P&L</th>
                            <th class="text-right py-2">P&L %</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            positions.forEach(pos => {
                const pnlClass = pos.unrealized_pnl >= 0 ? 'profit' : 'loss';
                html += `
                    <tr class="border-b">
                        <td class="py-2">${pos.symbol}</td>
                        <td class="py-2">${pos.side.toUpperCase()}</td>
                        <td class="text-right py-2">${pos.entry_price.toFixed(2)}</td>
                        <td class="text-right py-2">${pos.current_price.toFixed(2)}</td>
                        <td class="text-right py-2">${pos.size.toFixed(6)}</td>
                        <td class="text-right py-2 ${pnlClass}">
                            ${pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                        </td>
                        <td class="text-right py-2 ${pnlClass}">
                            ${pos.unrealized_pnl_pct >= 0 ? '+' : ''}${pos.unrealized_pnl_pct.toFixed(2)}%
                        </td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        
        function updateTrades(trades) {
            const container = document.getElementById('trades-table');
            
            if (trades.length === 0) {
                container.innerHTML = '<p class="text-gray-500">Aucun trade réalisé</p>';
                return;
            }
            
            let html = `
                <table class="min-w-full">
                    <thead>
                        <tr class="border-b">
                            <th class="text-left py-2">Paire</th>
                            <th class="text-right py-2">Trades</th>
                            <th class="text-right py-2">Gagnés</th>
                            <th class="text-right py-2">Perdus</th>
                            <th class="text-right py-2">P&L</th>
                            <th class="text-right py-2">Dernier Trade</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            trades.forEach(trade => {
                const pnlClass = trade.pnl >= 0 ? 'profit' : 'loss';
                const lastTrade = trade.last_trade ? new Date(trade.last_trade).toLocaleString() : '-';
                
                html += `
                    <tr class="border-b">
                        <td class="py-2">${trade.symbol}</td>
                        <td class="text-right py-2">${trade.trades}</td>
                        <td class="text-right py-2 text-green-600">${trade.wins}</td>
                        <td class="text-right py-2 text-red-600">${trade.losses}</td>
                        <td class="text-right py-2 ${pnlClass}">
                            ${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                        </td>
                        <td class="text-right py-2 text-sm text-gray-600">${lastTrade}</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        
        // Contrôles
        document.getElementById('pause-btn').addEventListener('click', async () => {
            try {
                const response = await fetch('/api/control/pause', { method: 'POST' });
                if (!response.ok) throw new Error(`Erreur HTTP: ${response.status}`);
                document.getElementById('pause-btn').classList.add('hidden');
                document.getElementById('resume-btn').classList.remove('hidden');
            } catch (error) {
                console.error('Erreur pause:', error);
            }
        });
        
        document.getElementById('resume-btn').addEventListener('click', async () => {
            try {
                const response = await fetch('/api/control/resume', { method: 'POST' });
                if (!response.ok) throw new Error(`Erreur HTTP: ${response.status}`);
                document.getElementById('resume-btn').classList.add('hidden');
                document.getElementById('pause-btn').classList.remove('hidden');
            } catch (error) {
                console.error('Erreur resume:', error);
            }
        });
        
        // Initialisation
        connectWebSocket();
        updateDashboard();
        setInterval(updateDashboard, 5000);  // Toutes les 5 secondes
        
        // Ping WebSocket pour maintenir la connexion
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 30000);  // Toutes les 30 secondes
    </script>
</body>
</html>
"""
    
    def set_bot(self, bot):
        """Définit le bot à monitorer"""
        self.bot = bot
    
    async def start_update_loop(self):
        """Démarre la boucle de mise à jour WebSocket"""
        while True:
            try:
                if self.bot and self.websocket_clients:
                    # Diffuser le status
                    await self.broadcast_update('status', self.bot.get_status())
                
                await asyncio.sleep(5)  # Mise à jour toutes les 5 secondes
                
            except Exception as e:
                log_error(f"Erreur dans update loop: {str(e)}")
                await asyncio.sleep(10)

dashboard_instance = Dashboard()
app = dashboard_instance.app