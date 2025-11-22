import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from src.config import PLOTS_PATH


def run_eda(df):
    print("\nЗапуск супер-красивого EDA...")

    os.makedirs(PLOTS_PATH, exist_ok=True)

    # Профессиональные настройки графиков
    plt.rcParams.update({
        'figure.figsize': (16, 9),
        'axes.titlesize': 20,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 14,
        'font.family': 'DejaVu Sans',
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'savefig.format': 'png'
    })
    sns.set_style("whitegrid")
    colors = ["#2c3e50", "#e74c3c", "#3498db", "#f39c12", "#1abc9c", "#9b59b6"]

    # 1. Распределение Market Cap
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Обычное распределение
    axes[0].hist(df['Market Cap'] / 1e9, bins=70, color=colors[2], alpha=0.9,
                 edgecolor='white', linewidth=1.2)
    axes[0].set_title('Распределение стоимости компании',
                      fontsize=22, fontweight='bold', pad=30)
    axes[0].set_xlabel('Капитализация, млрд $', fontsize=16)
    axes[0].set_ylabel('Количество компаний', fontsize=16)
    axes[0].grid(True, alpha=0.4)

    # После логарифмирования
    df['Market Cap_log'] = np.log1p(df['Market Cap'])        # ← правильная строка!
    axes[1].hist(df['Market Cap_log'], bins=70, color=colors[1], alpha=0.9,
                 edgecolor='white', linewidth=1.2)
    axes[1].set_title('После log1p-преобразования\n(целевая переменная)',
                      fontsize=22, fontweight='bold', pad=30)
    axes[1].set_xlabel('log1p(Market Cap)', fontsize=16)
    axes[1].set_ylabel('Количество компаний', fontsize=16)
    axes[1].grid(True, alpha=0.4)

    plt.suptitle('Анализ распределения целевой переменной',
                 fontsize=26, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, '01_market_cap_distribution.png'),
                facecolor='white')
    plt.close()

    # 2. Корреляционная матрица по МОДУЛЮ + маска
    cols = ['Price', 'Price/Earnings', 'EBITDA', 'Price/Sales', 'Price/Book', 'Market Cap_log']
    corr = df_corr = df[cols].corr().abs()  # ← ВОТ ЭТО ГЛАВНОЕ: .abs()!

    # Маска — убираем дублирование и диагональ
    mask = np.triu(np.ones_like(df_corr, dtype=bool))

    plt.figure(figsize=(13, 11))
    sns.heatmap(df_corr,
                mask=mask,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',      # яркая палитра
                center=0,
                vmin=0, vmax=1,       # от 0..1, потому что модуль
                square=True,
                linewidths=3,
                linecolor='white',
                cbar_kws={"shrink": 0.8, "label": "Абсолютная корреляция"},
                annot_kws={"size": 16, "weight": "bold", "color": "black"})

    plt.title('Топ корреляций с капитализацией\n(по модулю, без дублирования)',
              fontsize=24, fontweight='bold', pad=30)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, '02_correlation_matrix.png'),
                facecolor='white', dpi=400)
    plt.close()

    print(f"Красивые графики EDA сохранены в: {PLOTS_PATH}")