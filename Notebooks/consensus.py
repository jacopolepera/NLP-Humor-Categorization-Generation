import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, pearsonr

# === CONFIGURAZIONE ===

# File CSV per ciascun modello
data_files = {
    "llama3": "Classificazione IR/Sample 10000/categoriz_short_sample10000_llama3.csv",
    "mistral": "Classificazione IR/Sample 10000/categoriz_short_sample10000_mistral.csv",
    "gemma": "Classificazione IR/Sample 10000/categoriz_short_sample10000_gemma.csv",
    #"qmistral": "Classificazione IR/Sample 10000/categoriz_short_sample10000_qmistral.csv",
}

# Categorie da analizzare
categories = [
    "Edgy Content",
    "Cultural Reference",
    "Wordplay",
    "Absurdity",
    "Relatable",
    "Offensive Humor"
]

# === CARICAMENTO DATI ===

df_dict = {model: pd.read_csv(path) for model, path in data_files.items()}
dfs = list(df_dict.values())
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on="jokeText", how="inner")

# === ANALISI DI ACCORDO TRA MODELLI ===

agreement_results = []
model_pairs = list(combinations(data_files.keys(), 2))

for cat in categories:
    for m1, m2 in model_pairs:
        col1 = f"{m1}_{cat}"
        col2 = f"{m2}_{cat}"
        if col1 in merged_df.columns and col2 in merged_df.columns:
            sub_df = merged_df[[col1, col2]].dropna()
            values1, values2 = sub_df[col1], sub_df[col2]

            kappa = cohen_kappa_score(values1, values2)
            spearman, _ = spearmanr(values1, values2)
            pearson, _ = pearsonr(values1, values2)
            mad = (values1 - values2).abs().mean()
            strong_div = (values1 - values2).abs() >= 3

            agreement_results.append({
                "Categoria": cat,
                "Modelli": f"{m1} vs {m2}",
                "Cohen's Kappa": round(kappa, 3),
                "Spearman Corr": round(spearman, 3),
                "Pearson Corr": round(pearson, 3),
                "Media Differenze Assolute": round(mad, 3),
                "Divergenze Forti (>=3)": strong_div.sum()
            })

agreement_df = pd.DataFrame(agreement_results)
print("\n📊 Misure di accordo tra modelli per ciascuna categoria:")
print(agreement_df)

# === SCORE AGGREGATI ===

# Media, Mediana, Moda
for cat in categories:
    model_cols = [f"{model}_{cat}" for model in data_files.keys() if f"{model}_{cat}" in merged_df.columns]
    if len(model_cols) >= 2:
        merged_df[f"{cat}_mean"] = merged_df[model_cols].mean(axis=1)
        merged_df[f"{cat}_median"] = merged_df[model_cols].median(axis=1)
        merged_df[f"{cat}_mode"] = merged_df[model_cols].mode(axis=1)[0]  # prima moda

# === FUNZIONE LPR (score consensuale ponderato) ===

def compute_lpr(merged_df, category, weights=None, max_deviation=3):
    if weights is None:
        weights = {"llama3": 0.5, "mistral": 0.3, "gemma": 0.2}

    available_models = [model for model in weights if f"{model}_{category}" in merged_df.columns]
    usable_weights = {model: weights[model] for model in available_models}

    method_counts = {
        "media_ponderata_completa": 0,
        "media_ponderata_filtrata": 0,
        "fallback_llama3": 0
    }

    scores = []

    for _, row in merged_df.iterrows():
        ratings = {m: row[f"{m}_{category}"] for m in available_models}
        values = list(ratings.values())
        models = list(ratings.keys())

        # Calcola disaccordi tra le coppie
        pairs = [(m1, m2) for i, m1 in enumerate(models) for m2 in models[i+1:]]
        disagreements = {(m1, m2): abs(ratings[m1] - ratings[m2]) for m1, m2 in pairs}
        strong_disagreement = [pair for pair, dist in disagreements.items() if dist > max_deviation]

        if len(strong_disagreement) == 0:
            # Caso 1: consenso accettabile → media pesata
            method_counts["media_ponderata_completa"] += 1
            score = sum(ratings[m] * usable_weights[m] for m in models)
        else:
            # Caso 2: disaccordo → filtro sui modelli vicini alla media
            mean_value = np.mean(values)
            filtered_models = [m for m in models if abs(ratings[m] - mean_value) <= max_deviation + 1]
            if len(filtered_models) >= 2:
                method_counts["media_ponderata_filtrata"] += 1
                total_weight = sum(usable_weights[m] for m in filtered_models)
                score = sum(ratings[m] * usable_weights[m] for m in filtered_models) / total_weight
            else:
                # Caso 3: fallback → modello col peso maggiore
                method_counts["fallback_llama3"] += 1
                model_with_max_weight = max(usable_weights.items(), key=lambda x: x[1])[0]
                score = ratings[model_with_max_weight]
                
        scores.append(round(score))

    return pd.Series(scores, index=merged_df.index), method_counts

# === Calcolo LPR per tutte le categorie
global_method_counts = {}

for cat in categories:
    lpr_series, method_counts = compute_lpr(merged_df, cat)
    merged_df[f"{cat}_LPR"] = lpr_series
    global_method_counts[cat] = method_counts

# === Salvataggio file con nome contenente numero righe
lpr_cols = [f"{cat}_LPR" for cat in categories if f"{cat}_LPR" in merged_df.columns]
output_df = merged_df[["jokeText"] + lpr_cols]
n_rows = len(output_df)
output_filename = f"aggregated_scores_LPR_{n_rows}.csv"
output_df.to_csv(output_filename, index=False)
print(f"✅ File salvato come '{output_filename}'")

# === Report finale: conteggio modalità LPR per categoria
print("\n📊 Modalità di calcolo LPR per categoria:")
for cat, counts in global_method_counts.items():
    print(f"\n🧩 Categoria: {cat}")
    for method, count in counts.items():
        print(f"  - {method.replace('_', ' ').capitalize()}: {count}")
