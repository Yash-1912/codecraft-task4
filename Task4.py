import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_df = pd.read_csv("G:/Task4/twitter_training.csv", header=None)
valid_df = pd.read_csv("G:/Task4/twitter_validation.csv", header=None)

# Assign column names
train_df.columns = ['ID', 'Brand', 'Sentiment', 'Tweet']
valid_df.columns = ['ID', 'Brand', 'Sentiment', 'Tweet']

print("Training Dataset Shape:", train_df.shape)
print("Validation Dataset Shape:", valid_df.shape)

# Combine datasets
df = pd.concat([train_df, valid_df], ignore_index=True)

# Drop missing tweets
df.dropna(subset=['Tweet'], inplace=True)

# Sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Sentiment')
plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# Brand-wise sentiment analysis (Top 5 brands)
top_brands = df['Brand'].value_counts().head(5).index
brand_df = df[df['Brand'].isin(top_brands)]

plt.figure(figsize=(10,6))
sns.countplot(data=brand_df, x='Brand', hue='Sentiment')
plt.title("Brand-wise Sentiment Distribution")
plt.xlabel("Brand")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("brand_sentiment.png")
plt.show()

print("Analysis and visualization completed successfully.")