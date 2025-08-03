import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# === CONFIG ===
DATA_PATH = Path("results/exp0/tweets_with_finbert_sentiment.csv")
OUTPUT_DIR = Path("output/neutral_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
df['sentiment'] = df['sentiment'].str.lower()
df['finbert_label'] = df['finbert_label'].str.lower()

# === FILTER NEUTRAL PREDICTIONS ===
df_neutral = df[df['finbert_label'] == 'neutral']
print(f"Total tweets predicted as 'neutral': {len(df_neutral)}")

# === TRUE LABEL DISTRIBUTION ===
true_label_dist = df_neutral['sentiment'].value_counts(normalize=True)
true_label_dist.plot(kind='bar', color='gray')
plt.title("True Labels Distribution Among FinBERT Neutral Predictions")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "true_label_distribution.png")
plt.close()


# === NEUTRAL SCORE DISTRIBUTION ===
plt.figure(figsize=(8, 5))
sns.histplot(df_neutral['finbert_score'], bins=30, kde=True, color='blue')
plt.title("Distribution of FinBERT Scores (Neutral Predictions)")
plt.xlabel("FinBERT score")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "finbert_score_distribution.png")
plt.close()

# === WEEKLY NEUTRAL VOLUME ===
weekly = df_neutral.groupby('week').size()
weekly.plot(kind='bar', color='gray', figsize=(14, 5))
plt.title("Weekly Volume of Neutral Predictions")
plt.ylabel("Tweet count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "neutral_volume_by_week.png")
plt.close()

from sklearn.metrics import classification_report

# === RELABELING STRATEGY ===
positive_threshold = 0.2
negative_threshold = -0.2

# Apply relabel strategy (corrected version)
df_neutral = df_neutral.copy()  # avoid SettingWithCopyWarning
df_neutral['relabel'] = df_neutral['finbert_score'].apply(
    lambda score: "positive" if score > positive_threshold
    else "negative" if score < negative_threshold
    else "neutral"
)

# Count tweets per relabel suggestion
relabel_summary = {
    "Reclassify as positive (score > 0.2)": (df_neutral['relabel'] == "positive").sum(),
    "Reclassify as negative (score < -0.2)": (df_neutral['relabel'] == "negative").sum(),
    "Remain neutral": (df_neutral['relabel'] == "neutral").sum()
}
relabel_df = pd.DataFrame.from_dict(relabel_summary, orient='index', columns=["Tweet count"])

# === EVALUATE RELABELING STRATEGY (EXCLUDING 'neutral') ===
df_eval = df_neutral[df_neutral['relabel'].isin(['positive', 'negative'])]
true_labels = df_eval['sentiment']
pred_labels = df_eval['relabel']

# Import classification_report and generate summary
report_text = classification_report(true_labels, pred_labels, digits=3)

# === PLOT WITH TEXT EMBEDDED IN IMAGE ===
plt.figure(figsize=(10, 6))
ax = relabel_df.plot(kind='barh', color='seagreen', legend=False, ax=plt.gca())
plt.title("Suggested Post-Labeling Strategy for Neutral Predictions")
plt.xlabel("Tweet Count")

# Embed classification report in the image
plt.gcf().text(0.58, 0.15, report_text, fontsize=8, family='monospace', verticalalignment='bottom')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "relabel_suggestion.png", dpi=300)
plt.close()




# === FINAL SUMMARY ===
print("\n ANALYSIS SUMMARY")
print(f"✔ True label distribution for 'neutral':\n{true_label_dist.round(3)}")
# =Random samples for manual inspection =
print("\n Example tweets predicted as 'neutral':")
print(df_neutral[['date', 'ticker', 'text', 'sentiment', 'finbert_score']].sample(5, random_state=42).to_string(index=False))
