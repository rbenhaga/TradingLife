# Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/yourusername/crypto-trading-bot.git
   cd crypto-trading-bot
   ```

2. Créez un environnement virtuel et installez les dépendances :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. Copiez le fichier `.env.example` puis renseignez vos clés API :
   ```bash
   cp .env.example .env
   # Éditer .env
   ```

4. Lancez les tests pour vérifier l'installation :
   ```bash
   python run_tests.py
   ```
