#!/usr/bin/env python3
"""
Script de test de connexion à l'API Binance
Vérifie que les clés API fonctionnent et affiche les informations de base
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import ccxt
from colorama import init, Fore, Style

# Initialiser colorama pour les couleurs dans le terminal
init()

def print_success(message):
    """Affiche un message de succès en vert"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    """Affiche un message d'erreur en rouge"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_info(message):
    """Affiche un message d'information en bleu"""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")

def print_warning(message):
    """Affiche un message d'avertissement en jaune"""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")

def main():
    """Fonction principale du test de connexion"""
    
    print(f"\n{Fore.CYAN}{'='*50}")
    print("🚀 TEST DE CONNEXION API BINANCE")
    print(f"{'='*50}{Style.RESET_ALL}\n")
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Vérifier la présence des clés API
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    testnet = os.getenv('TESTNET', 'true').lower() == 'true'
    
    if not api_key or not api_secret:
        print_error("Les clés API ne sont pas configurées!")
        print_info("Veuillez configurer BINANCE_API_KEY et BINANCE_API_SECRET dans le fichier .env")
        return 1
    
    print_success("Clés API trouvées dans l'environnement")
    print_info(f"Mode: {'TESTNET' if testnet else 'PRODUCTION'}")
    
    # Afficher les premières lettres des clés pour vérification
    print_info(f"API Key commence par: {api_key[:8]}...")
    print_info(f"API Secret commence par: {api_secret[:8]}...")
    
    try:
        # Initialiser l'exchange
        if testnet:
            # Configuration spéciale pour le testnet Binance
            exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'createMarketBuyOrderRequiresPrice': False,
                }
            })
            
            # Configuration manuelle du testnet
            exchange.set_sandbox_mode(True)
            exchange.urls['api'] = {
                'public': 'https://testnet.binance.vision/api/v3',
                'private': 'https://testnet.binance.vision/api/v3',
                'v1': 'https://testnet.binance.vision/api/v1'
            }
            
            print_info("Configuration testnet spot appliquée")
            
            # Test 1: Vérifier le statut de l'exchange
            print(f"\n{Fore.YELLOW}📡 Test de connexion...{Style.RESET_ALL}")
            try:
                # Test avec une requête publique simple d'abord
                print_info("Test de connexion de base...")
                time = exchange.fetch_time()
                print_success("Connexion de base réussie")
                
                # Test avec une requête de ticker directe
                print_info("Tentative de récupération du ticker BTC/USDT...")
                response = exchange.publicGetTickerPrice({'symbol': 'BTCUSDT'})
                price = float(response['price'])
                print_success("Connexion publique réussie")
                print(f"   Prix BTC/USDT: ${price:,.2f}")
                
            except ccxt.NetworkError as e:
                print_error(f"Erreur réseau: {str(e)}")
                print_info("Détails de l'erreur:")
                print_info(f"  - Type: {type(e).__name__}")
                print_info(f"  - Message: {str(e)}")
                print_info("Vérifiez votre connexion Internet")
                return 1
            except ccxt.ExchangeError as e:
                print_error(f"Erreur d'échange: {str(e)}")
                print_info("Détails de l'erreur:")
                print_info(f"  - Type: {type(e).__name__}")
                print_info(f"  - Message: {str(e)}")
                if "fapiPublic" in str(e):
                    print_info("Erreur liée aux futures détectée. Vérifiez la configuration du testnet.")
                return 1
            except KeyError as e:
                print_error(f"Erreur de configuration: {str(e)}")
                print_info("Détails de l'erreur:")
                print_info(f"  - Type: {type(e).__name__}")
                print_info(f"  - Message: {str(e)}")
                print_info("Problème de configuration du testnet")
                return 1
            except Exception as e:
                print_error(f"Erreur inattendue: {str(e)}")
                print_info("Détails de l'erreur:")
                print_info(f"  - Type: {type(e).__name__}")
                print_info(f"  - Message: {str(e)}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                }
            })
            print_warning("Connexion à Binance PRODUCTION!")
        
        # Test 2: Récupérer l'heure du serveur
        print(f"\n{Fore.YELLOW}🕐 Synchronisation temporelle:{Style.RESET_ALL}")
        try:
            time = exchange.fetch_time()
            server_time = datetime.fromtimestamp(time / 1000)
            local_time = datetime.now()
            time_diff = abs((server_time - local_time).total_seconds())
            
            print(f"   Heure serveur:  {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Heure locale:   {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Différence:     {time_diff:.2f} secondes")
            
            if time_diff > 1:
                print_warning("La différence de temps est importante. Assurez-vous que votre horloge est synchronisée.")
            else:
                print_success("Synchronisation temporelle OK")
        except Exception as e:
            print_warning(f"Impossible de récupérer l'heure serveur: {str(e)}")
        
        # Test 3: Vérifier le compte et les balances
        print(f"\n{Fore.YELLOW}💰 Vérification du compte...{Style.RESET_ALL}")
        try:
            # Récupérer les balances directement
            response = exchange.privateGetAccount()
            balances = response.get('balances', [])
            
            # Filtrer les balances non nulles
            non_zero_balances = [
                balance for balance in balances 
                if float(balance.get('free', 0)) > 0 or float(balance.get('locked', 0)) > 0
            ]
            
            if non_zero_balances:
                print_success(f"Authentification réussie! Fonds trouvés ({len(non_zero_balances)} actifs)")
                print(f"\n{Fore.YELLOW}💵 Balances du compte:{Style.RESET_ALL}")
                for balance in sorted(non_zero_balances, key=lambda x: x['asset'])[:10]:
                    asset = balance['asset']
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked
                    print(f"   - {asset}: {total:,.4f} (disponible: {free:,.4f})")
                    
                if len(non_zero_balances) > 10:
                    print(f"   ... et {len(non_zero_balances) - 10} autres actifs")
            else:
                print_warning("Authentification réussie mais aucun fonds trouvé")
                print_info("Les fonds testnet sont normalement attribués automatiquement")
                print_info("Vérifiez sur https://testnet.binance.vision/")
                
        except ccxt.AuthenticationError as e:
            print_error(f"Erreur d'authentification: {str(e)}")
            print_info("Vérifiez que vos clés API sont correctes et actives")
            print_info("Pour le testnet, assurez-vous d'utiliser les clés du testnet")
            return 1
        except Exception as e:
            print_error(f"Erreur lors de la vérification du compte: {str(e)}")
            return 1
        
        # Test 4: Tester la récupération de données OHLCV
        print(f"\n{Fore.YELLOW}📈 Test de récupération des données historiques...{Style.RESET_ALL}")
        try:
            # Utiliser l'endpoint spot pour les données OHLCV
            response = exchange.publicGetKlines({
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'limit': 5
            })
            
            if response:
                print_success(f"Dernières {len(response)} bougies horaires récupérées")
                
                # Afficher la dernière bougie
                last_candle = response[-1]
                # Convertir le timestamp en entier avant la division
                timestamp = int(last_candle[0])
                candle_time = datetime.fromtimestamp(timestamp / 1000)
                
                # Convertir toutes les valeurs en float
                open_price = float(last_candle[1])
                high_price = float(last_candle[2])
                low_price = float(last_candle[3])
                close_price = float(last_candle[4])
                volume = float(last_candle[5])
                
                print(f"   Dernière bougie ({candle_time.strftime('%H:%M')}):")
                print(f"   - Open:  ${open_price:,.2f}")
                print(f"   - High:  ${high_price:,.2f}")
                print(f"   - Low:   ${low_price:,.2f}")
                print(f"   - Close: ${close_price:,.2f}")
                print(f"   - Volume: {volume:,.2f} BTC")
            else:
                print_warning("Aucune donnée OHLCV récupérée")
        except Exception as e:
            print_warning(f"Impossible de récupérer les données OHLCV: {str(e)}")
            print_info("Détails de l'erreur:")
            print_info(f"  - Type: {type(e).__name__}")
            print_info(f"  - Message: {str(e)}")
        
        # Résumé final
        print(f"\n{Fore.GREEN}{'='*50}")
        print("✅ TEST DE CONNEXION RÉUSSI!")
        print(f"{'='*50}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}📝 Prochaines étapes:{Style.RESET_ALL}")
        if testnet:
            print("1. Vos fonds testnet devraient être visibles ci-dessus")
            print("2. Si ce n'est pas le cas, connectez-vous sur https://testnet.binance.vision/")
        print("3. Configurez vos paramètres de trading dans config/settings.py")
        print("4. Lancez le bot avec: python src/main.py --paper")
        
        return 0
        
    except ccxt.NetworkError as e:
        print_error(f"Erreur réseau: {str(e)}")
        print_info("Vérifiez votre connexion Internet")
        return 1
    except Exception as e:
        print_error(f"Erreur inattendue: {str(e)}")
        print_info(f"Type d'erreur: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
