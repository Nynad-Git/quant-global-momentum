# Changelog – Quant Global Momentum Project

## Baseline V2 – Initial Version
- Long-only momentum strategy
- Volatility and liquidity filters
- Inverse-volatility weighting
- Monthly rebalancing

Performance:
- Sharpe ≈ 0.91
- Max drawdown ≈ -22.8%

---

## Baseline V2 – Weight Cap Added
**Change**
- Added 10% maximum weight cap per stock

**Motivation**
- Reduce concentration risk
- Improve robustness during market stress

**Result**
- Sharpe improved to ≈ 0.96
- Turnover slightly reduced
- Drawdown slightly worsened

---

## Baseline V2 – Weight Smoothening
**Change**
- 80% old stock weightage, 20% new

**Motivation**
- Reduce concentration risk
- Improve robustness during market stress

**Result**
- Sharpe improved to ≈ 1.05
- Turnover reduced ~ 0.145
- Drawdown slightly worsened

---
