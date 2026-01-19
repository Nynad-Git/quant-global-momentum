# Quantitative Momentum & ML Strategy (US Equities)

This project implements and compares rule-based momentum strategies and a machine-learning-driven (Ridge regression) strategy on US equities, evaluated against the S&P 500 (SPY) benchmark.

---

## Project Structure & Execution Flow

### 1. Data Collection

**File to run**
```bash
python src/fetch_data.py
````

**What it does**

* Downloads historical OHLCV data for the US equity universe
* Cleans and aligns time series data

**Outputs**

* `data/raw/us_ohlcv.parquet`

---

### 2. Baseline Momentum Strategy (V2)

**File to run**

```bash
python src/backtest_baseline_v2.py
```

**What it does**

* Long-only momentum strategy
* Volatility and liquidity filtering
* Inverse-volatility position sizing
* Monthly rebalancing with transaction cost modeling

**Outputs**

* `results/baseline_v2_us_equity_curve.csv`
* Console performance metrics (return, volatility, Sharpe ratio, drawdown, turnover)

---

### 3. S&P 500 Benchmark (SPY)

**File to run**

```bash
python src/benchmark_spy.py
```

**What it does**

* Computes buy-and-hold performance of SPY
* Aligns benchmark returns to strategy rebalance dates

**Outputs**

* `results/spy_benchmark_equity_curve.csv`
* Console benchmark performance metrics

---

### 4. Baseline vs SPY Comparison Report

**File to run**

```bash
python src/report_us_baseline_vs_spy.py
```

**What it does**

* Compares Baseline V2 strategy against SPY
* Generates equity and drawdown visualizations

**Outputs**

* `results/equity_curve_baseline_v2_vs_spy.png`
* `results/drawdown_baseline_v2_vs_spy.png`
* `results/metrics_baseline_v2_vs_spy.csv`

---

### 5. Machine Learning Strategy (Ridge Regression)

**File to run**

```bash
python src/ml_ridge_strategy.py
```

**What it does**

* Rolling out-of-sample Ridge (L2) regression
* Predicts forward returns using engineered momentum features
* Risk and liquidity filtering
* Inverse-volatility weighting and turnover control

**Outputs**

* `results/ml_ridge_us_equity_curve.csv`
* Console ML strategy performance metrics

---

## Key Results (Summary)

* Baseline V2 achieves risk-adjusted performance comparable to SPY with controlled drawdowns
* ML Ridge strategy demonstrates improved return potential with higher complexity and model risk
* Full results are reproducible using the scripts above

```
