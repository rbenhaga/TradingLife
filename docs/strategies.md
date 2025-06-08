# Stratégies

## MultiSignalStrategy

Cette stratégie regroupe plusieurs indicateurs : RSI, MACD, bandes de Bollinger,
volume et croisements de moyennes mobiles. Chaque indicateur produit un signal
entre -1 et 1. Le `WeightedScoreEngine` applique des poids à ces signaux pour
obtenir un score global.

Lorsque le score dépasse un seuil positif, la stratégie émet un ordre d'achat.
En dessous d'un seuil négatif, elle recommande la vente. Les poids des indicateurs
peuvent être ajustés dans le code pour affiner le comportement.
