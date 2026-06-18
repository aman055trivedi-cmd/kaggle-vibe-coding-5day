# Day 3 — End-to-End ML Pipeline with Model Comparison
# Kaggle 5-Day Vibe Coding Program
# Author: Aman Trivedi
# Day 3 was my favorite — building a proper pipeline felt really satisfying!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings; warnings.filterwarnings('ignore')

print("🤖 Day 3 — End-to-End ML Pipeline")
print("=" * 50)

np.random.seed(42)
n = 1200
X = np.random.randn(n, 10)
y = ((X[:,0] + X[:,1]*2 - X[:,2] + np.random.randn(n)*0.5) > 0).astype(int)

cols = [f'feature_{i}' for i in range(10)]
df = pd.DataFrame(X, columns=cols)
df['target'] = y

X_train, X_test, y_train, y_test = train_test_split(df[cols], df['target'], test_size=0.2, random_state=42)

pipelines = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(random_state=42))]),
    'Random Forest':        Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(n_estimators=100, random_state=42))]),
    'Gradient Boosting':    Pipeline([('scaler', StandardScaler()), ('model', GradientBoostingClassifier(random_state=42))]),
    'SVM':                  Pipeline([('scaler', StandardScaler()), ('model', SVC(probability=True, random_state=42))]),
    'XGBoost':              Pipeline([('scaler', StandardScaler()), ('model', XGBClassifier(random_state=42, verbosity=0))]),
}

results = []
print("\nTraining pipelines...")
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc').mean()
    test = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    results.append({'Model': name, 'CV AUC': round(cv,4), 'Test AUC': round(test,4)})
    print(f"  {name:25s} CV AUC: {cv:.4f} | Test AUC: {test:.4f}")

results_df = pd.DataFrame(results).sort_values('Test AUC', ascending=False)
print(f"\n🏆 Best model: {results_df.iloc[0]['Model']} (AUC: {results_df.iloc[0]['Test AUC']})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Day 3 — ML Pipeline Results', fontsize=14, fontweight='bold', color='#1B3A6B')

axes[0].barh(results_df['Model'], results_df['Test AUC'], color='#1B3A6B', alpha=0.85)
axes[0].set_xlim(0.5, 1.0); axes[0].set_title('Test AUC by Model'); axes[0].set_xlabel('AUC Score')
for i, v in enumerate(results_df['Test AUC']): axes[0].text(v+0.002, i, f'{v:.4f}', va='center', fontsize=9)

best_pipe = pipelines[results_df.iloc[0]['Model']]
fpr, tpr, _ = roc_curve(y_test, best_pipe.predict_proba(X_test)[:,1])
axes[1].plot(fpr, tpr, color='#1B3A6B', lw=2, label=f'AUC = {results_df.iloc[0]["Test AUC"]:.4f}')
axes[1].plot([0,1],[0,1],'r--', lw=1); axes[1].set_title('ROC Curve — Best Model')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].legend()

plt.tight_layout()
plt.savefig('day3_pipeline_results.png', dpi=150)
plt.show()
print("\n✅ Day 3 Complete!")
