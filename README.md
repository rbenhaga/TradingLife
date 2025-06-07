# Crypto Trading Bot

Un algorithme de trading automatisé pour les cryptomonnaies, conçu pour fonctionner 24/7 sur un VPS.

## 🚀 Caractéristiques

- ✅ Trading automatisé 24/7
- ✅ Support multi-exchanges (Binance, Bybit)
- ✅ Gestion du risque intégrée
- ✅ Backtesting sur données historiques
- ✅ Interface web de monitoring
- ✅ Architecture modulaire et extensible

## 📋 Prérequis

- Python 3.9+
- Compte Binance avec API activée
- VPS Linux (Oracle Free Tier compatible)

## 🛠️ Installation

```bash
# Cloner le repository
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer l'environnement
cp .env.example .env
# Éditer .env avec vos clés API
⚙️ Configuration

Créer un compte Binance et générer des clés API
Configurer les clés dans .env
Ajuster les paramètres de risque selon votre profil

🚦 Utilisation
bash# Test de connexion
python scripts/test_connection.py

# Lancer le bot (paper trading)
python src/main.py --paper

# Lancer le bot (trading réel)
python src/main.py --live
📊 Stratégies
Le bot implémente plusieurs stratégies :

Moving Average Crossover (MA)
RSI Oversold/Overbought
MACD Momentum
Custom strategies (extensible)

🔒 Sécurité

Clés API avec permissions limitées (pas de retrait)
Stop-loss automatique sur chaque position
Limite de drawdown journalier
Logs détaillés de toutes les opérations

📝 License
MIT License - Voir LICENSE pour plus de détails.
🤝 Contribution
Les contributions sont les bienvenues ! Voir CONTRIBUTING.md pour les guidelines.

### 5. `setup.py`
```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crypto-trading-bot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Un algorithme de trading automatisé pour les cryptomonnaies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto-trading-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "ccxt>=4.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "sqlalchemy>=2.0.0",
        "fastapi>=0.104.0",
        "pandas-ta>=0.3.14b0",
    ],
    entry_points={
        "console_scripts": [
            "cryptobot=src.main:main",
        ],
    },
)