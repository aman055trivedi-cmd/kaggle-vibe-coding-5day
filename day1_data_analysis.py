# Day 1 — Exploratory Data Analysis with AI Assistance
# Kaggle 5-Day Vibe Coding Program
# Author: Aman Trivedi
# Day 1 was all about using AI tools to speed up EDA. Pretty cool experience!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings_off = __import__('warnings'); warnings_off.filterwarnings('ignore')

print("🚀 Day 1 — AI-Assisted Exploratory Data Analysis")
print("=" * 50)

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'age':          np.random.randint(18, 70, n),
    'income':       np.random.normal(50000, 15000, n).clip(15000, 150000),
    'education':    np.random.choice(['High School','Bachelor','Master','PhD'], n, p=[0.3,0.4,0.2,0.1]),
    'job_type':     np.random.choice(['Tech','Finance','Healthcare','Education','Other'], n),
    'experience':   np.random.randint(0, 40, n),
    'satisfaction': np.random.randint(1, 11, n),
})
df['income'] = df['income'] + df['experience'] * 500 + np.random.normal(0, 5000, n)

print(f"\n📊 Dataset shape: {df.shape}")
print(f"\n🔍 Basic Info:")
print(df.describe().round(2))
print(f"\n❓ Missing values: {df.isnull().sum().sum()}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Day 1 — EDA Dashboard', fontsize=16, fontweight='bold', color='#1B3A6B')

axes[0,0].hist(df['income'], bins=40, color='#1B3A6B', alpha=0.8, edgecolor='white')
axes[0,0].set_title('Income Distribution'); axes[0,0].set_xlabel('Income ($)')

axes[0,1].scatter(df['experience'], df['income'], alpha=0.3, color='coral', s=15)
axes[0,1].set_title('Experience vs Income')

edu_income = df.groupby('education')['income'].mean().sort_values()
axes[0,2].barh(edu_income.index, edu_income.values, color='steelblue')
axes[0,2].set_title('Avg Income by Education')

sns.boxplot(data=df, x='job_type', y='satisfaction', ax=axes[1,0], palette='Blues')
axes[1,0].set_title('Job Satisfaction by Type')
axes[1,0].tick_params(axis='x', rotation=30)

corr_cols = ['age','income','experience','satisfaction']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,1])
axes[1,1].set_title('Correlation Heatmap')

df['age_group'] = pd.cut(df['age'], bins=[18,30,45,60,70], labels=['18-30','31-45','46-60','60+'])
age_counts = df['age_group'].value_counts().sort_index()
axes[1,2].pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', colors=plt.cm.Blues(np.linspace(0.3,0.9,4)))
axes[1,2].set_title('Age Group Distribution')

plt.tight_layout()
plt.savefig('day1_eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Day 1 Complete! EDA dashboard saved.")
