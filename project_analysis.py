import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
from scipy.stats import mannwhitneyu, pearsonr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

# --- 1. CONFIGURATION & REPRODUCIBILITY ---
NP_SEED = 42
np.random.seed(NP_SEED)

# Increased pool for more robust sampling across multiple events
POOL_SIZE = 3000   
SAMPLE_PER_GROUP = 400  # N=400 before, N=400 after for each event
BATCH_SIZE = 32
OUTPUT_DIR = 'outputs'

EVENTS = {
    "WHO_Pandemic": pd.to_datetime('2020-03-11', utc=True),
    "Fed_Unlimited_QE": pd.to_datetime('2020-03-23', utc=True)
}

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)
nltk.download('vader_lexicon', quiet=True)

def fast_cliffs_delta(x, y):
    """Vectorized Cliff's Delta: O(n^2) but optimized via NumPy."""
    x, y = np.asarray(x), np.asarray(y)
    return (np.sum(x[:, None] > y[None, :]) - np.sum(x[:, None] < y[None, :])) / (len(x) * len(y))

# --- 2. DATA LOADING & FILTERING ---
print("--- LOADING DATA ---")
df = pd.read_csv('raw_analyst_ratings.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df = df.dropna(subset=['date', 'headline'])

# Filter for the relevant COVID-19 window
df = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2020-05-31')].copy()

# Create a pool to score once (Research-grade caching)
df_pool = df.sample(min(len(df), POOL_SIZE), random_state=NP_SEED).copy()

# --- 3. SENTIMENT INFERENCE (POOL CACHING) ---
print(f"--- SCORING POOL (N={len(df_pool)}) ---")

# Try to use Apple Silicon GPU (device=0), fallback to CPU (device=-1)
try:
    finbert = pipeline("text-classification", model="ProsusAI/finbert", truncation=True, device=0)
    print("Using MPS (GPU) acceleration.")
except Exception:
    finbert = pipeline("text-classification", model="ProsusAI/finbert", truncation=True, device=-1)
    print("Using CPU for inference.")

headlines = df_pool['headline'].astype(str).tolist()
results = finbert(headlines, batch_size=BATCH_SIZE)

def parse_finbert(res):
    score = res['score']
    if res['label'] == 'positive': return score
    if res['label'] == 'negative': return -score
    return 0.0

df_pool['finbert_score'] = [parse_finbert(r) for r in results]

vader = SentimentIntensityAnalyzer()
df_pool['vader_score'] = df_pool['headline'].apply(lambda x: vader.polarity_scores(x)['compound'])

# --- 4. MULTI-EVENT BALANCED ANALYSIS ---
print("--- RUNNING STATISTICAL TESTS ---")
summary_list = []

for event_name, ts in EVENTS.items():
    # Balanced groups: same number of samples before and after the event
    before_pool = df_pool[df_pool['date'] < ts]
    after_pool = df_pool[df_pool['date'] >= ts]
    
    n = min(SAMPLE_PER_GROUP, len(before_pool), len(after_pool))
    
    b_sample = before_pool.sample(n, random_state=NP_SEED)
    a_sample = after_pool.sample(n, random_state=NP_SEED)
    
    # Statistics
    x = b_sample['finbert_score'].values
    y = a_sample['finbert_score'].values
    
    stat, p_val = mannwhitneyu(x, y, alternative="two-sided")
    delta = fast_cliffs_delta(y, x) # Effect of moving from before to after
    
    summary_list.append({
        "Event": event_name,
        "N_Per_Group": n,
        "Mean_Before": x.mean(),
        "Mean_After": y.mean(),
        "MW_P_Value": p_val,
        "Cliffs_Delta": delta
    })
    
    # Save the sampled data for reproducibility check
    sampled = pd.concat([b_sample, a_sample])
    sampled['Period'] = np.where(sampled['date'] < ts, 'Before', 'After')
    sampled.to_csv(f"{OUTPUT_DIR}/sampled_{event_name}.csv", index=False)

    # Visualization: Violin Plots
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=sampled, x='Period', y='finbert_score', hue='Period', 
                   dodge=False, legend=False, palette='coolwarm', inner='quart')
    plt.title(f"{event_name.replace('_', ' ')} Sentiment Shift\nDelta: {delta:.3f}, p: {p_val:.4f}")
    plt.savefig(f"{OUTPUT_DIR}/violin_{event_name}.png")
    plt.close()

# Save final summary table
summary_df = pd.DataFrame(summary_list)
summary_df.to_csv(f"{OUTPUT_DIR}/event_comparison_summary.csv", index=False)
print(summary_df)

# --- 5. MARKET VALIDATION ---
print("--- RUNNING MARKET CORRELATION ---")
market_data = yf.download("^GSPC", start="2020-01-01", end="2020-06-01", progress=False)

# Fix MultiIndex if it exists
if isinstance(market_data.columns, pd.MultiIndex):
    market_data.columns = market_data.columns.get_level_values(0)

market_data['Return'] = market_data['Close'].pct_change()
market_data = market_data.reset_index()
market_data['Date'] = market_data['Date'].dt.date

# Aggregate daily sentiment
daily_sentiment = df_pool.groupby(df_pool['date'].dt.date)['finbert_score'].mean().reset_index()
daily_sentiment.columns = ["Date", "Avg_Sentiment"]

merged = pd.merge(daily_sentiment, market_data[['Date', 'Close', 'Return']], on='Date', how='inner').dropna()
corr, corr_p = pearsonr(merged["Avg_Sentiment"], merged["Return"])

# Visualizing Market vs Sentiment
fig, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('S&P 500 Close', color=color, fontweight='bold')
ax1.plot(merged['Date'], merged['Close'], color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('News Sentiment (7-Day Rolling)', color=color, fontweight='bold')
merged['Rolling_Sent'] = merged['Avg_Sentiment'].rolling(7).mean()
ax2.plot(merged['Date'], merged['Rolling_Sent'], color=color, linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

# Mark the WHO event
plt.axvline(EVENTS["WHO_Pandemic"].date(), color='red', linestyle=':', label='WHO Pandemic')
plt.title(f"S&P 500 vs FinBERT Sentiment (Corr: {corr:.3f}, p: {corr_p:.3f})")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/market_vs_sentiment.png")
plt.close()

print(f"\nFinal Correlation: {corr:.4f} (p={corr_p:.4f})")
print(f"All outputs saved to the '{OUTPUT_DIR}' folder.")