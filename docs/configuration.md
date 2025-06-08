# Configuration

Le fichier `config/config.json` définit les paramètres du bot :

* `exchange`
  - `name` : nom de l'exchange utilisé (ex. `binance`).
  - `testnet` : active le mode test de l'exchange.
  - `api_key` / `api_secret` : vos clés API.

* `trading`
  - `pairs` : liste des paires traitées.
  - `initial_balance` : capital initial pour les simulations.
  - `position_size` : part du capital utilisée par trade.
  - `max_positions` : nombre maximal de positions ouvertes.
  - `stop_loss` / `take_profit` : seuils par défaut appliqués aux ordres.

* `strategy`
  - `name` : nom de la stratégie active.
  - `short_window` et `long_window` : périodes des moyennes mobiles.
  - `trend_window` : période de détection de tendance.
  - `timeframe` : intervalle de temps des bougies.
  - `min_data_points` : nombre minimal de données avant de lancer la stratégie.

* `risk_management`
  - `max_drawdown` : perte maximale autorisée sur le capital.
  - `daily_loss_limit` : perte maximale par jour.
  - `position_sizing` : méthode de calcul de la taille de position.

* `logging`
  - `level` : niveau de verbosité des logs.
  - `file` : fichier dans lequel les logs sont écrits.
