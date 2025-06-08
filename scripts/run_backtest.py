"""Script de backtest complet"""

import argparse
import pandas as pd
import ccxt

from src.core.backtester import Backtester
from src.strategies.multi_signal import MultiSignalStrategy
from src.core.logger import log_info, log_error


def load_historical_data(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Charge les données historiques via CCXT"""
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Lance un backtest sur des données historiques")
    parser.add_argument("--symbol", default="BTC/USDT", help="Paire à trader")
    parser.add_argument("--timeframe", default="1h", help="Timeframe des données")
    parser.add_argument("--limit", type=int, default=500, help="Nombre de bougies")
    parser.add_argument("--engine", choices=["custom", "vectorbt"], default="vectorbt", help="Moteur de simulation")
    parser.add_argument("--parallel", action="store_true", help="Exécution parallèle avec Ray")
    args = parser.parse_args()

    try:
        data = load_historical_data(args.symbol, args.timeframe, args.limit)
        strategy = MultiSignalStrategy(args.symbol, timeframe=args.timeframe)
        backtester = Backtester(strategy, engine=args.engine)

        if args.parallel:
            results = backtester.run_parallel(data, [{}])
            for i, res in enumerate(results):
                log_info(f"Run {i} -> Return {res.total_return_pct:.2f}%")
        else:
            result = backtester.run(data)
            log_info(f"Return: {result.total_return_pct:.2f}%")
    except Exception as e:
        log_error(f"Erreur lors du backtest: {e}")


if __name__ == "__main__":
    main()
    asyncio.run(main())
