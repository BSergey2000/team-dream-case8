# Создаёт два графика для EDA и сохраняет в results/plots


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .config import DATA_PATH, PLOTS_DIR


def generate_eda_plots():
    # Читаем данные из файла
    df = pd.read_csv(DATA_PATH)

    # Убираем строки без капитализации и отрицательные значения
    df = df.dropna(subset=["Market Cap"])
    df = df[df["Market Cap"] > 0].copy()

    # График 01 — показывает, почему нужен логарифм
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.hist(df["Market Cap"] / 1e9, bins=60, color="#1f77b4", edgecolor="white")
    ax1.set_title("Распределение Market Cap")
    ax1.set_xlabel("млрд $")
    ax1.set_ylabel("компаний")

    log_cap = np.log1p(df["Market Cap"])
    ax2.hist(log_cap, bins=60, color="#d62728", edgecolor="white")
    ax2.set_title("log(Market Cap + 1)")
    ax2.set_xlabel("логарифм капитализации")
    ax2.set_ylabel("компаний")

    plt.suptitle("Распределение рыночной капитализации")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_eda_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()  # закрываем, чтобы память не засорялась

    # График 02 — корреляция между всеми числовыми признаками
    cols = ["Price", "Price/Earnings", "Dividend Yield", "Earnings/Share",
            "52 Week Low", "52 Week High", "Market Cap", "EBITDA",
            "Price/Sales", "Price/Book"]

    corr = df[cols].corr().round(2)  # считаем корреляцию
    mask = np.triu(np.ones_like(corr, dtype=bool))  # скрываем верхний треугольник

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr.abs(), mask=mask, annot=True, fmt=".2f", cmap="Blues",
                linewidths=1, linecolor='white', cbar_kws={"shrink": 0.8})
    plt.title("Корреляция признаков (|ρ|)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_eda_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()