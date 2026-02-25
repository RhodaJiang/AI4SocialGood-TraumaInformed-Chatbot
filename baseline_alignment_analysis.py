import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# =============================
# 0) CONFIG — 改成你的真实文件名
# =============================
CHATGPT_XLSX = "chatgpt_BERT.xlsx"          # <- 改
MISTRAL_XLSX = "mistral_BERT.xlsx"        # <- 改
BASELINE_XLSX = "senarios for SBERT analysis.xlsx"  # 你的baseline表

BASELINE_TEXT_COL = "outputs from chatgpt-5.2"  # baseline文本列名（按你文件）
K = 5  # top-k mean: 推荐 3 或 5


# =============================
# Helpers
# =============================
def _normalize_iteration_to_num(series: pd.Series) -> pd.Series:
    """
    Accepts iteration values like:
    - 0,1,2,3,4
    - 1,2,3,4,5
    - v0,v1,... or V0
    - iter0, iteration0, version0
    Returns numeric iter_num (float), NaN if cannot parse.
    """
    s = series.astype(str).str.strip().str.lower()
    s = (
        s.replace({"nan": ""})
         .str.replace("iteration", "", regex=False)
         .str.replace("version", "", regex=False)
         .str.replace("iter", "", regex=False)
         .str.replace("v", "", regex=False)
         .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")


def _load_outputs(path: str, expected_cols=None) -> pd.DataFrame:
    df = pd.read_excel(path)
    expected_cols = expected_cols or {"scenario", "input_text", "output_text", "model", "iteration"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Output file '{path}' missing columns: {missing}")

    # clean
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["iteration"] = df["iteration"].astype(str).str.strip()
    df["output_text"] = df["output_text"].fillna("").astype(str)

    df["iter_num"] = _normalize_iteration_to_num(df["iteration"])
    if df["iter_num"].isna().any():
        bad = df.loc[df["iter_num"].isna(), "iteration"].dropna().unique()[:20]
        raise ValueError(
            f"Cannot parse some iteration values in '{path}'. Examples: {bad}. "
            f"Please standardize iteration to e.g. v0-v4 or 0-4."
        )

    return df


# =============================
# 1) Load 2 model outputs + concat
# =============================
df_chatgpt = _load_outputs(CHATGPT_XLSX)
df_mistral = _load_outputs(MISTRAL_XLSX)

df = pd.concat([df_chatgpt, df_mistral], ignore_index=True)

print("Total outputs rows:", len(df))
print("Rows by model:\n", df["model"].value_counts(dropna=False))


# =============================
# 2) Load baseline exemplars (20 texts)
# =============================
baseline_df = pd.read_excel(BASELINE_XLSX)
if BASELINE_TEXT_COL not in baseline_df.columns:
    raise ValueError(
        f"Baseline file '{BASELINE_XLSX}' missing column '{BASELINE_TEXT_COL}'. "
        f"Available columns: {list(baseline_df.columns)}"
    )

baseline_texts = baseline_df[BASELINE_TEXT_COL].fillna("").astype(str).tolist()

# 若你 baseline 表里除了20条还有别的空行/备注，这里可以过滤空文本
baseline_texts = [t.strip() for t in baseline_texts if t and t.strip()]

if len(baseline_texts) < 1:
    raise ValueError("No valid baseline texts found after cleaning.")

print("Baseline exemplar count:", len(baseline_texts))


# =============================
# 3) SBERT similarity: each output vs baseline set
# =============================
sbert = SentenceTransformer("all-mpnet-base-v2")

resp_embed = sbert.encode(
    df["output_text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

ref_embed = sbert.encode(
    baseline_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

sim = cos_sim(resp_embed, ref_embed).cpu().numpy()  # shape: [N, B]

df["baseline_maxsim"] = sim.max(axis=1)

# Top-k mean (稳健推荐)
k = int(min(K, sim.shape[1]))
df["baseline_topkmean"] = np.sort(sim, axis=1)[:, -k:].mean(axis=1)

# （可选）你也可以保留 mean_all 做对照，但一般不推荐用于主结论
df["baseline_meanall"] = sim.mean(axis=1)


# =============================
# 4) Summary by model & iteration
# =============================
summary = (
    df.groupby(["model", "iter_num"])["baseline_topkmean"]
      .agg(["mean", "median", "std", "count"])
      .reset_index()
      .sort_values(["model", "iter_num"])
)

print("\nTop-k mean cosine to expert baseline set — summary:")
print(summary)


# =============================
# 5) Mixed Effects Model: random intercept by scenario
# =============================
df["model"] = df["model"].astype("category")

print("\nRunning MixedLM (random intercept: scenario)...")
mixed = smf.mixedlm(
    "baseline_topkmean ~ iter_num * model",
    df,
    groups=df["scenario"]
).fit()

print(mixed.summary())


# =============================
# 6) Plot trend (both models on one figure)
# =============================
plt.figure()
for m in df["model"].cat.categories:
    sub = summary[summary["model"] == m]
    plt.plot(sub["iter_num"], sub["mean"], marker="o", label=str(m))

iters = sorted(df["iter_num"].unique())
plt.xticks(iters, [f"v{int(i)}" for i in iters])
plt.xlabel("Iteration")
plt.ylabel(f"Top-{k} mean cosine similarity to expert baselines")
plt.title("Expert-baseline semantic alignment across iterations")
plt.legend()
plt.tight_layout()
plt.savefig("expert_baseline_alignment.png", dpi=200)
print("\nSaved: expert_baseline_alignment.png")


# =============================
# 7) Save scored table + summary
# =============================
df.to_csv("expert_baseline_similarity_scored.csv", index=False)
summary.to_csv("expert_baseline_summary.csv", index=False)

print("Saved: expert_baseline_similarity_scored.csv")
print("Saved: expert_baseline_summary.csv")