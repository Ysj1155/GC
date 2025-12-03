import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/cota_tune_small/summary.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    # 1) COTA 튜닝 결과만 필터
    cond = (
        (df["policy"] == "cota") &
        (df["ops"] == 200000) &
        (df["update_ratio"] == 0.8) &
        (df["hot_ratio"] == 0.2)
    )
    df_cota = df[cond].copy()

    # 혹시 컬럼 이름이 다르면 print(df.columns) 찍어보고 이름 맞춰주면 됨

    # 2) (alpha, gamma) 조합별로 WAF / wear_std 요약
    group_cols = ["cota_alpha", "cota_gamma"]  # beta, delta는 고정값이니까 생략 가능
    agg = (
        df_cota
        .groupby(group_cols)
        .agg(
            waf_mean=("waf", "mean"),
            waf_std=("waf", "std"),
            wear_mean=("wear_std", "mean"),
            wear_std=("wear_std", "std"),
            n=("seed", "count"),
        )
        .reset_index()
    )

    print("=== COTA 하이퍼파라미터별 요약 ===")
    print(agg)

    # 필요하면 논문/부록용 테이블로 저장
    agg.to_csv("results/cota_tune_small/summary_by_param.csv", index=False)

    # 3) WAF–wear_std 산점도 (파레토 느낌)
    fig, ax = plt.subplots()
    for _, row in agg.iterrows():
        x = row["waf_mean"]
        y = row["wear_mean"]
        label = f"α={row.cota_alpha}, γ={row.cota_gamma}"
        ax.scatter(x, y)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)

    ax.set_xlabel("WAF (mean)")
    ax.set_ylabel("wear_std (mean)")
    ax.set_title("COTA Hyperparameter Sweep (ops=200k, update_ratio=0.8)")
    plt.tight_layout()
    plt.savefig("results/cota_tune_small/cota_pareto_scatter.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()