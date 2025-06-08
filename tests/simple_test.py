#!/usr/bin/env python3
"""
Script pour vérifier la configuration CCXT et les URLs de Binance
"""

import ccxt
import json

def main():
    print("🔍 VÉRIFICATION CONFIGURATION CCXT BINANCE")
    print("=" * 50)
    
    # Vérifier la version de CCXT
    print(f"📦 Version CCXT: {ccxt.__version__}")
    
    # Créer une instance Binance
    exchange = ccxt.binance()
    
    # Afficher les URLs par défaut
    print(f"\n🌐 URLs par défaut:")
    print(json.dumps(exchange.urls, indent=2))
    
    # Vérifier les URLs de test/sandbox
    print(f"\n🧪 Support Testnet:")
    print(f"Has sandbox: {exchange.has.get('sandbox', False)}")
    print(f"Test URL: {exchange.urls.get('test', 'Non définie')}")
    
    # Configuration pour testnet
    print(f"\n⚙️  Configuration pour testnet:")
    testnet_exchange = ccxt.binance({
        'sandbox': True,
        'enableRateLimit': True,
    })
    
    print(f"URLs après configuration sandbox:")
    print(json.dumps(testnet_exchange.urls, indent=2))
    
    # Vérifier les endpoints disponibles
    print(f"\n📡 Endpoints disponibles:")
    print(f"Public: {testnet_exchange.has.get('fetchTicker', False)}")
    print(f"Private: {testnet_exchange.has.get('fetchBalance', False)}")
    print(f"Trading: {testnet_exchange.has.get('createOrder', False)}")
    
    # Recommandations
    print(f"\n💡 Recommandations:")
    print("1. Utiliser sandbox=True pour le testnet")
    print("2. Forcer defaultType='spot' pour éviter les futures")
    print("3. Augmenter le timeout si nécessaire")
    print("4. Vérifier que les clés API sont bien du testnet")

if __name__ == "__main__":
    main()
