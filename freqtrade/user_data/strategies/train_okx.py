from pathlib import Path
import pandas as pd
from GKD_FisherTransformV4_ML import GKD_FisherTransformV4_ML, MLOptimizer

# Список пар
pairs = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT", 
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "ADA/USDT:USDT",
    "XRP/USDT:USDT"
]

timeframe = "15m"
strategy = GKD_FisherTransformV4_ML()

for pair in pairs:
    print(f"\n🎯 Training pair: {pair}")

    # Формируем путь к файлу
    base, quote = pair.split("/")[0], pair.split("/")[1].replace(":", "_")
    data_file = Path(f"user_data/data/okx/futures/{base}_{quote}-{timeframe}-futures.feather")

    if not data_file.exists():
        print(f"❌ File not found for {pair}: {data_file}")
        continue

    df = pd.read_feather(data_file)
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert(None)

    # Разделение на train (например, последние 1000 свечей)
    df_train = df.tail(1000)

    optimizer_name = f"fisher_transform_v4_{base}_{quote}_{timeframe}"
    strategy.ml_optimizers[pair] = MLOptimizer(optimizer_name)

    # Обучаем
    strategy.perform_startup_training(df_train, pair)
    print(f"✅ ML trained for {pair}")
