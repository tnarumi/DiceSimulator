import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import kstest, norm


# Simulation utilities
def roll_sample_means(
    sample_size: int, sample_count: int, seed: int | None, sides: int = 6, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Return sample means for rolling `sample_count` samples of `sample_size` dice."""
    rng = rng or np.random.default_rng(seed)
    rolls = rng.integers(1, sides + 1, size=(sample_count, sample_size))
    return rolls.mean(axis=1)


def cumulative_means(means: np.ndarray) -> np.ndarray:
    cumsum = np.cumsum(means, dtype=float)
    idx = np.arange(1, len(means) + 1)
    return cumsum / idx


def plot_cumulative_mean_of_rolls(rolls: np.ndarray) -> io.BytesIO:
    """Plot cumulative mean as sample size (number of rolls) grows."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(rolls) + 1), cumulative_means(rolls), color="#2ca02c", linewidth=1.5)
    ax.axhline(3.5, color="#d62728", linestyle="--", linewidth=1, label="True mean (3.5)")
    ax.set_xlabel("Number of rolls (sample size)")
    ax.set_ylabel("Cumulative mean of rolls")
    ax.set_title("Law of Large Numbers (by sample size)")
    ax.set_ylim(1, 6)
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


# Plot helpers
def plot_histogram(
    means: np.ndarray,
    sample_size: int,
    sides: int = 6,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> io.BytesIO:
    """Hist of sample means with normal approximation overlay; returns PNG bytes."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(means, bins=30, density=True, alpha=0.7, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Sample mean")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of sample means")

    # Normal approximation (CLT): dice have mean mu and variance sigma^2.
    mu = (1 + sides) / 2
    var = (sides**2 - 1) / 12
    sigma = (var / sample_size) ** 0.5
    xs = np.linspace(means.min(), means.max(), 200)
    ax.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), color="#d62728", linewidth=2, label="Normal approx")
    ax.legend()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


# Streamlit UI
st.set_page_config(page_title="Dice Sample Distribution", page_icon="🎲", layout="wide")
st.title("Dice Sample Distribution (CLT / LLN demo)")
st.write(
    " サンプル数 m とサンプルサイズ n とを指定してサイコロの標本平均の分布を可視化します．"
    " 標本平均のヒストグラムに加えて，サンプルサイズを増やしたときの累積平均がどう収束するかを示すグラフも描画します．"
)

with st.sidebar:
    st.header("Simulation settings")
    m = st.number_input("Number of samples; m", min_value=1, max_value=1_000_000, value=1000, step=100)
    n = st.number_input("Sample size; n", min_value=1, max_value=100_000, value=10, step=1)
    st.header("Plot options")
    manual_x = st.checkbox("ヒストグラム x 軸を手動設定", value=False)
    if manual_x:
        hist_x_min = st.number_input("x min", value=1.0, step=0.1)
        hist_x_max = st.number_input("x max", value=6.0, step=0.1)
        hist_x_lim = (hist_x_min, hist_x_max)
    else:
        hist_x_lim = (1.0, 6.0)
    manual_y = st.checkbox("ヒストグラム y 軸を手動設定", value=False)
    if manual_y:
        hist_y_min = st.number_input("y min", value=0.0, step=0.1)
        hist_y_max = st.number_input("y max", value=1.0, step=0.1)
        hist_y_lim = (hist_y_min, hist_y_max)
    else:
        hist_y_lim = None
    st.header("Simulation options")
    use_seed = st.checkbox("乱数シードを指定する", value=False)
    seed_opt = st.text_input("Random seed (整数)", value="") if use_seed else ""

# Input validation
validation_errors: list[str] = []
seed = None
if use_seed:
    if not seed_opt.strip():
        validation_errors.append("Seed を入力してください。")
    else:
        try:
            seed = int(seed_opt)
        except ValueError:
            validation_errors.append("Seed は整数を入力してください。")

if hist_x_lim and hist_x_lim[0] >= hist_x_lim[1]:
    validation_errors.append("x 軸の下限は上限より小さい値にしてください。")
if hist_y_lim and hist_y_lim[0] >= hist_y_lim[1]:
    validation_errors.append("y 軸の下限は上限より小さい値にしてください。")
if int(n) * int(m) > 1e8:
    validation_errors.append("計算量が大きすぎます（n×m ≤ 1e8 にしてください）。")

# Run simulation lazily, only if inputs are valid
if validation_errors:
    for err in validation_errors:
        st.error(err)
    st.session_state.means = None
    st.session_state.rolls_for_lln = None
else:
    with st.spinner("Simulating dice rolls..."):
        rng = np.random.default_rng(seed)
        st.session_state.means = roll_sample_means(int(n), int(m), seed, rng=rng)
        st.session_state.rolls_for_lln = rng.integers(1, 7, size=int(n))
        st.session_state.last_run = time.strftime("%Y-%m-%d %H:%M:%S")

means = st.session_state.get("means")
rolls_for_lln = st.session_state.get("rolls_for_lln")

if means is None or rolls_for_lln is None:
    st.info("左の設定で「Run simulation」を押してください。")
else:
    st.caption(f"Last run: {st.session_state.get('last_run', '')}  |  seed = {seed if seed_opt else 'random'}")

    hist_png = plot_histogram(means, sample_size=int(n), xlim=hist_x_lim, ylim=hist_y_lim)
    lln_png = plot_cumulative_mean_of_rolls(rolls_for_lln)
    hist_name = f"sampleDist_n{int(n)}_m{int(m)}.png"
    run_name = f"cumulativeMean_n{int(n)}.png"

    # CLT goodness-of-fit (KS test vs normal approximation)
    mu = (1 + 6) / 2
    var = (6**2 - 1) / 12
    sigma = (var / int(n)) ** 0.5
    ks_stat, ks_p = kstest(means, "norm", args=(mu, sigma))

    col1, col2 = st.columns(2)
    with st.container():
        st.subheader("中心極限定理の一致度 (KS検定)")
        met1, met2, met3 = st.columns(3)
        met1.metric("Sample size n", f"{int(n):,}")
        met2.metric("KS statistic", f"{ks_stat:.4f}")
        met3.metric("p-value", f"{ks_p:.4f}")
        st.markdown(
            "KS statistic は標本分布と理論正規分布の最大乖離の大きさで、0 に近いほど差が小さいです。"
            " p-value は「正規分布と矛盾すると言えるほど差が大きいか」の指標で、小さいほど正規近似からの逸脱を示唆し、大きいほど正規近似と矛盾しない（このサンプルでは棄却できない）ことを意味します。"
            " 例として 0.05 未満だと『正規近似と矛盾する差があるかも』と判断しやすく、0.05 以上なら『このサンプル量では正規近似と矛盾しない』と読むのが一般的です（閾値設定は文脈依存です）。"
            " 一般に p 値は小さいほど「珍しい」ため良いと誤解されがちですが、ここでは近似の良さを見るので p が大きい ≒ 正規近似と矛盾しない、p が小さい ≒ 正規近似からずれている、という向きになります。"
        )

    with col1:
        st.subheader("Sample distribution")
        st.image(hist_png, caption="")
        st.download_button(
            "Download sample distribution (PNG)",
            data=hist_png.getvalue(),
            file_name=hist_name,
            mime="image/png",
        )
    with col2:
        st.subheader("Cumulative mean")
        st.image(lln_png, caption="")
        st.download_button(
            "Download cumulative mean (PNG)",
            data=lln_png.getvalue(),
            file_name=run_name,
            mime="image/png",
        )

    st.markdown("---")
    st.markdown(
        "- ヒストグラムは中心極限定理によりサンプルサイズ n が大きいほど正規分布に近づきます。\n"
        "- 累積平均のグラフは、サンプルサイズ（ロール数）を増やしたときに平均が 3.5 に収束する様子を示しています。\n"
        "- seed を指定すると再現可能なシミュレーションになります。"
    )
