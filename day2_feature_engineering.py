# Day 2 — Feature Engineering & Data Preprocessing
# Kaggle 5-Day Vibe Coding Program
# Author: Aman Trivedi
# Feature engineering is honestly an art — Day 2 taught me that!

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

print("🔧 Day 2 — Feature Engineering & Preprocessing")
print("=" * 50)

np.random.seed(42)
n = 800
df = pd.DataFrame({
    'age':         np.random.randint(18, 65, n),
    'salary':      np.random.normal(60000, 20000, n).clip(20000, 200000),
    'tenure':      np.random.randint(0, 20, n),
    'dept':        np.random.choice(['Tech','HR','Finance','Sales','Ops'], n),
    'perf_score':  np.random.uniform(1, 5, n),
    'overtime_hrs':np.random.randint(0, 50, n),
    'dist_km':     np.random.uniform(1, 100, n),
})
df['left'] = ((df['perf_score'] < 2.5) | (df['overtime_hrs'] > 35) | (df['salary'] < 35000)).astype(int)

print(f"Raw features: {df.shape[1]-1}")
print(f"Target distribution:\n{df['left'].value_counts()}")

# Feature Engineering
df['salary_per_year']   = df['salary'] / (df['tenure'] + 1)
df['work_life_score']   = df['perf_score'] / (df['overtime_hrs'] + 1) * 10
df['age_tenure_ratio']  = df['age'] / (df['tenure'] + 1)
df['high_performer']    = (df['perf_score'] > 3.5).astype(int)
df['long_commute']      = (df['dist_km'] > 50).astype(int)
df['salary_band']       = pd.qcut(df['salary'], q=4, labels=['Low','Mid','High','VHigh'])

le = LabelEncoder()
df['dept_enc']        = le.fit_transform(df['dept'])
df['salary_band_enc'] = le.fit_transform(df['salary_band'])

feature_cols = ['age','salary','tenure','perf_score','overtime_hrs','dist_km',
                'salary_per_year','work_life_score','age_tenure_ratio',
                'high_performer','long_commute','dept_enc','salary_band_enc']

X = df[feature_cols]; y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare raw vs engineered features
rf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
rf_raw.fit(X_train[['age','salary','tenure','perf_score','overtime_hrs','dist_km']], y_train)
acc_raw = accuracy_score(y_test, rf_raw.predict(X_test[['age','salary','tenure','perf_score','overtime_hrs','dist_km']]))

rf_eng = RandomForestClassifier(n_estimators=100, random_state=42)
rf_eng.fit(X_train, y_train)
acc_eng = accuracy_score(y_test, rf_eng.predict(X_test))

print(f"\n📈 Raw features accuracy    : {acc_raw:.4f}")
print(f"📈 Engineered features accuracy: {acc_eng:.4f}")
print(f"🚀 Improvement: +{(acc_eng-acc_raw)*100:.2f}%")

importances = pd.Series(rf_eng.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(10, 7))
colors = ['#1B3A6B' if i >= len(importances)-5 else '#aabbd4' for i in range(len(importances))]
importances.plot(kind='barh', color=colors)
plt.title('Feature Importance — Engineered Features', fontsize=13, fontweight='bold', color='#1B3A6B')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('day2_feature_importance.png', dpi=150)
plt.show()
print("\n✅ Day 2 Complete!")
