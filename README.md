# Crypto Trading Bot 🤖

Bot de trading automatisé pour cryptomonnaies avec support multi-paires et stratégies avancées.

## 📋 Fonctionnalités

- ✅ Trading automatisé 24/7
- ✅ Support multi-paires avec sélection intelligente
- ✅ Stratégies basées sur indicateurs techniques
- ✅ Gestion des risques (stop-loss, take-profit)
- ✅ Mode paper trading pour tests
- ✅ Interface web de monitoring
- ✅ Backtesting intégré
- ✅ Optimisation des paramètres avec Optuna

## 🚀 Installation

```bash
# Cloner le repository
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API
```

## ⚙️ Configuration

1. **Clés API Exchange** : Ajoutez vos clés dans `.env`
2. **Paramètres de trading** : Modifiez `config/config.json`
3. **Règles de trading** : Ajustez `config/trading_rules.py`

## 📖 Usage

### Mode Paper Trading (Test)
```bash
python run_bot.py --paper
```

### Mode Trading Réel
```bash
python run_bot.py --real
```

### Lancer un Backtest
```bash
python scripts/run_backtest.py --symbol BTC/USDT --days 30
```

### Optimiser les Paramètres
```bash
python scripts/optimize_weights.py --symbol BTC/USDT --trials 100
```

### Interface Web
```bash
python src/web/dashboard.py
# Ouvrir http://localhost:5000
```

## 📊 Stratégies Disponibles

- **Multi-Signal** : Combine RSI, MACD, Bollinger Bands, Volume
- **Score Pondéré** : Système de scoring avec poids optimisables
- **Volatility Scanner** : Sélection automatique des meilleures paires

## 🛡️ Sécurité

- Ne jamais partager vos clés API
- Utiliser des clés avec permissions limitées (pas de retrait)
- Activer la whitelist IP sur l'exchange
- Stocker les clés dans des variables d'environnement

## 📈 Performance

Les performances dépendent de nombreux facteurs :
- Conditions de marché
- Paramètres de la stratégie
- Gestion du risque
- Frais de trading

**Avertissement** : Le trading de cryptomonnaies comporte des risques. Ce bot est fourni à titre éducatif.

## 🤝 Contribution

Les contributions sont bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE)

## 🔧 Support

- Documentation : [docs/](docs/)
- Issues : [GitHub Issues](https://github.com/yourusername/crypto-trading-bot/issues)
- Discord : [Rejoindre le serveur](https://discord.gg/xxxxx)

## 🎯 Roadmap

- [ ] Support multi-exchanges
- [ ] Intégration IA/ML
- [ ] Trading futures
- [ ] Application mobile
- [ ] Notifications Telegram

---

Développé avec ❤️ pour la communauté crypto