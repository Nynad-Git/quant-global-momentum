import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")

ML_FILE = RESULTS / "ml_ridge_us_equity_curve.csv"
BASE_FILE = RESULTS / "baseline_v2_us_equity_curve.csv"
SPY_FILE = RESULTS / "spy_benchmark_equity_curve.csv"

OUT_EQUITY = RESULTS / "equity_curve_ml_vs_baseline_vs_spy.png"
OUT_DD = RESULTS / "drawdown_ml_vs_baseline_vs_spy.png"


def load_curve(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "equity" not in df.columns:
        raise ValueError(f"{path} must contain columns: date, equity")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")[["date", "equity"]].copy()
    df = df.rename(columns={"equity": name}).set_index("date")
    return df


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def main():
    for p in [ML_FILE, BASE_FILE, SPY_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    ml = load_curve(ML_FILE, "ML Ridge")
    base = load_curve(BASE_FILE, "Baseline V2")
    spy = load_curve(SPY_FILE, "SPY")

    # Align on common dates
    curves = ml.join(base, how="inner").join(spy, how="inner")

    # Equity curve plot
    plt.figure()
    curves.plot()
    plt.title("Equity Curve: ML Ridge vs Baseline V2 vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Equity (Growth of $1)")
    plt.tight_layout()
    plt.savefig(OUT_EQUITY, dpi=160)
    plt.close()

    # Drawdown plot
    dd = curves.apply(drawdown)
    plt.figure()
    dd.plot()
    plt.title("Drawdown: ML Ridge vs Baseline V2 vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(OUT_DD, dpi=160)
    plt.close()

    print("Saved:")
    print(f"- {OUT_EQUITY}")
    print(f"- {OUT_DD}")
    print("\nLatest equity values (aligned dates):")
    print(curves.iloc[-1].to_string())


if __name__ == "__main__":
    main()
