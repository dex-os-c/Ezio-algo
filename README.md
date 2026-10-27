# algo_ai: SMC / ICT + auction-profile crypto model with a paper-trading web app

Paper trading only. No exchange keys are used and no orders are ever sent.

## Start
```bash
pip install -r requirements.txt
python server.py            # open http://localhost:8000
```
In the app: pick BTC/USDT or ETH/USDT, press **Train model**, read the report, then use the **Chart and trading** tab to run a paper session.
Tick "fake test data" first if you only want to check that everything runs without downloading candles.

On a cloud GPU box: `HOST=0.0.0.0 PORT=8000 python server.py` and open the forwarded port. Torch uses CUDA automatically.

## What it does
1. **Features** (`smc.py`, `amt.py`, `features.py`), all causal. `python test_features.py` proves that no feature changes when future candles are added.
   - **SMC / ICT**: confirmed swing highs and lows (known only after 10 bars), BOS and CHoCH, fair value gaps, order blocks, liquidity sweeps, premium / discount and OTE zones, displacement candles, killzones in New York time, Asian range and previous-day high / low raids, 4h and 1d structure bias.
   - **Auction Market Theory**: rolling volume profile (POC, value area high and low), balance versus trend, value migration, failed auctions at the value-area edges, volume-delta proxy, rejection tails, previous-day value area.
2. **Model** (`models.py`): small GRU (GPU when available) plus LightGBM, averaged, predicting whether price is higher `--horizon` bars ahead.
3. **Honest testing** (`train.py`): purged walk-forward folds, then a final holdout used once, with fees and slippage. It prints a verdict and shows whether the model leans on SMC, auction or plain features.
4. **Paper trading** (`engine.py`): simulated broker with a risk layer (stop loss, daily loss halt, max drawdown halt, kill switch).
   - **Replay**: runs the untouched holdout period fast, so you can test in minutes instead of weeks.
   - **Live**: closed candles from the exchange. Between candles it marks your position to the live price.
   - **Manual orders**: buy, sell, close from the UI.
   - **Keep training**: every N bars it fine-tunes a copy from the current weights and promotes it only if it beats the live model on bars neither trained on. Live sessions resume from `model_latest`; replay always starts from the baseline so it stays honest.

## Command line
```bash
python train.py --symbol BTC/USDT --timeframe 1h --years 8 --out runs/btc
python train.py --symbol ETH/USDT --timeframe 15m --years 3 --features smc,amt --out runs/eth
python paper.py --out runs/btc --mode replay --speed 0
python paper.py --out runs/btc --continual true          # live
touch runs/btc/STOP                                       # kill switch
```

## About "20 years"
Crypto does not have 20 years of good data. Binance starts in 2017 (about 8 to 9 years). For the longest BTC history use
`--exchange bitstamp --symbol BTC/USD --timeframe 1d --years 15` (daily bars only, so killzones and intraday auction features are not used).
More years also means more regimes (2017 mania, 2018 bear, 2020 crash, 2021 top, 2022 bear), which is good for honest validation.
If Binance is blocked from your server, use `--exchange bybit` or `okx`.

## What is simplified (be aware)
- Real Market Profile uses TPO letters or tick data. Here every candle's volume is spread over its high-low range. Crypto volume is also exchange-specific.
- SMC and ICT are discretionary. The code uses mechanical definitions (listed above), so it is *a* reading of those concepts, not *the* reading. Timeframe, swing length and displacement thresholds are in `smc.py`.
- Not implemented: time-and-price theory such as IPDA ranges and power-of-3 timing, open interest, funding rates, order book.
- Paper fills happen at the last close with slippage. Real fills in fast markets are worse.

## Read this before trusting any number
- Most models like this find no edge after fees. The report says so plainly. On random fake data the correct answer is "no edge" and that is what you get.
- Look at the holdout once. Tuning against it makes it useless.
- Run paper trading for weeks across different market conditions before thinking about real money. Going live (exchange API, order handling, leverage, tax and legal rules) is deliberately not included.
- Crypto perps are leveraged. This code uses no leverage (size is a share of equity).
