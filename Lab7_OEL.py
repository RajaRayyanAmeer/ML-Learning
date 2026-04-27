import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 100

# Step 1: Loading Data
df = pd.read_csv('Lab 7 + OEL IBM HR Analytics Employee Attrition.csv')

print("Dataset Shape:", df.shape)
print("\nMissing Values:", df.isnull().sum().sum())
print("\nAttrition Distribution:\n", df['Attrition'].value_counts())

# Calculate attrition rate manually
total_employees = len(df)
left_employees = len(df[df['Attrition'] == 'Yes'])
attrition_rate = (left_employees / total_employees) * 100
print(f"\nAttrition Rate: {attrition_rate:.1f}%")

# Separate columns by type
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Step 2: Data Visualization
colors = ['#4C72B0', '#DD8452']

# Figure 1: Attrition Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

counts = df['Attrition'].value_counts()

bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', width=0.5)

for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15, str(val), ha='center', fontweight='bold', fontsize=12)

axes[0].set_title('Attrition Count', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Employees')

pie_labels = []
for label, val in zip(counts.index, counts.values):
    pct = (val / total_employees) * 100
    pie_labels.append(f'{label}\n({pct:.1f}%)')

axes[1].pie(counts.values, labels=pie_labels, colors=colors, startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1].set_title('Attrition Proportion', fontsize=13, fontweight='bold')

plt.suptitle('Figure 1: Overall Attrition Overview', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Figure 2: Department-wise Attrition
dept_attr = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
dept_rate_values = {}
for dept in dept_attr.index:
    yes_count   = dept_attr.loc[dept, 'Yes']
    total_count = dept_attr.loc[dept].sum()
    dept_rate_values[dept] = (yes_count / total_count) * 100

dept_rate = pd.Series(dept_rate_values).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dept_attr.plot(kind='bar', ax=axes[0], color=colors, edgecolor='white', width=0.6)
axes[0].set_title('Attrition Count by Department', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)
for container in axes[0].containers:
    axes[0].bar_label(container, fontsize=9, padding=2)

rate_bars = axes[1].bar(dept_rate.index, dept_rate.values, color=['#e74c3c', '#e67e22', '#27ae60'], edgecolor='white', width=0.5)

for bar, val in zip(rate_bars, dept_rate.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)

axes[1].set_title('Attrition Rate (%) by Department', fontsize=13, fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)
axes[1].set_ylim(0, 25)

plt.suptitle('Figure 2: Department-wise Attrition', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Figure 3: Salary vs Attrition
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df.boxplot(column='MonthlyIncome', by='Attrition', ax=axes[0], medianprops=dict(color='red', linewidth=2))

axes[0].set_title('Monthly Income by Attrition', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Attrition')
axes[0].set_ylabel('Monthly Income ($)')

for status, color in zip(['Yes', 'No'], colors[::-1]):
    subset = df[df['Attrition'] == status]['MonthlyIncome']
    axes[1].hist(subset, bins=30, alpha=0.55, color=color, label=f'Attrition = {status}', edgecolor='white')

axes[1].set_title('Income Distribution by Attrition', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Monthly Income ($)')
axes[1].legend()

plt.suptitle('Figure 3: Salary vs Attrition', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Figure 4: Work-Life Balance Violin
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.violinplot(x='Attrition', y='WorkLifeBalance', data=df, ax=axes[0], palette={'No': '#4C72B0', 'Yes': '#DD8452'}, inner='box', linewidth=1.2)

axes[0].set_title('Work-Life Balance by Attrition', fontsize=13, fontweight='bold')
axes[0].set_yticks([1, 2, 3, 4])
axes[0].set_yticklabels(['Bad', 'Good', 'Better', 'Best'])

wlb_levels = sorted(df['WorkLifeBalance'].unique())
wlb_rate_values = {}
for level in wlb_levels:
    subset = df[df['WorkLifeBalance'] == level]
    left   = len(subset[subset['Attrition'] == 'Yes'])
    total  = len(subset)
    wlb_rate_values[level] = (left / total) * 100

wlb_labels = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
wlb_x      = [wlb_labels[lvl] for lvl in wlb_rate_values.keys()]
wlb_y      = list(wlb_rate_values.values())

axes[1].bar(wlb_x, wlb_y, color=['#e74c3c', '#e67e22', '#27ae60', '#2980b9'], edgecolor='white', width=0.5)

for i, val in enumerate(wlb_y):
    axes[1].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)

axes[1].set_title('Attrition Rate by WLB Level', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 35)

plt.suptitle('Figure 4: Work-Life Balance Impact', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Figure 5: Years at Company
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for status, color in zip(['Yes', 'No'], colors[::-1]):
    subset = df[df['Attrition'] == status]['YearsAtCompany']
    axes[0].hist(subset, bins=20, alpha=0.6, color=color, label=f'Attrition = {status}', edgecolor='white')

axes[0].set_title('Years at Company Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Years at Company')
axes[0].legend()

df['Tenure_Group'] = pd.cut(df['YearsAtCompany'], bins=[0, 2, 5, 10, 20, 40], labels=['0-2', '3-5', '6-10', '11-20', '21+'])

tenure_groups = ['0-2', '3-5', '6-10', '11-20', '21+']
tenure_rate_values = {}
for group in tenure_groups:
    subset = df[df['Tenure_Group'] == group]
    left = len(subset[subset['Attrition'] == 'Yes'])
    total = len(subset)
    if total > 0:
        tenure_rate_values[group] = (left / total) * 100
    else:
        tenure_rate_values[group] = 0

bars = axes[1].bar(list(tenure_rate_values.keys()), list(tenure_rate_values.values()), color=['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#2980b9'], edgecolor='white', width=0.5)

for bar, val in zip(bars, tenure_rate_values.values()):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)

axes[1].set_title('Attrition Rate by Tenure Group', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 45)

plt.suptitle('Figure 5: Tenure vs Attrition', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

df.drop(columns=['Tenure_Group'], inplace=True)

# Step 3: Data Preprocessing
df_model = df.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])

# Convert target: Yes -> 1, No -> 0
df_model = df_model.copy()
attrition_encoded = []
for value in df_model['Attrition']:
    if value == 'Yes':
        attrition_encoded.append(1)
    else:
        attrition_encoded.append(0)
df_model['Attrition'] = attrition_encoded

# One-hot encode all remaining categorical columns
cat_features = df_model.select_dtypes(include='object').columns.tolist()
df_encoded = pd.get_dummies(df_model, columns=cat_features, drop_first=True)

print(f"\nShape after encoding: {df_encoded.shape}")

# Separate features (X) and target (y)
X = df_encoded.drop(columns=['Attrition'])
y = df_encoded['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples | Features: {X_train.shape[1]}")

# Step 4: Model Training
models = {
    'Logistic Regression': LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nCross-Validation ROC-AUC")
for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"  {name:25s}: {scores.mean():.4f} +/- {scores.std():.4f}")

# Train each model on the full training set
fitted_models = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    fitted_models[name] = model
print("\nAll models trained.")

# Step 5: Evaluation
palette = ['#4C72B0', '#DD8452']

results = []
for name, model in fitted_models.items():
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_prob)
    rep  = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        'Model':     name,
        'Accuracy':  round(acc, 4),
        'ROC-AUC':   round(roc, 4),
        'Precision': round(rep['1']['precision'], 4)
    })

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
print("\nFinal Model Comparison")
print(results_df.to_string(index=False))

# Figure 6: Model Comparison Bar Chart
metrics = ['Accuracy', 'ROC-AUC', 'Precision']
x     = np.arange(len(results_df))
width = 0.25
metric_palette = ['#4C72B0', '#DD8452', '#DD4567']

fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, results_df[metric], width, label=metric, color=metric_palette[i], edgecolor='white', alpha=0.88)

ax.set_xticks(x + width)
ax.set_xticklabels(results_df['Model'], rotation=10, ha='right', fontsize=10)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.set_title('Figure 6: Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.show()

# Figure 7: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, model) in zip(axes, fitted_models.items()):
    y_pred = model.predict(X_test_sc)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(cm, display_labels=['Stay', 'Leave'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=10, fontweight='bold')

plt.suptitle('Figure 7: Confusion Matrices', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Figure 8: ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
for (name, model), color in zip(fitted_models.items(), palette):
    y_prob    = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc_score:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Figure 8: ROC Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.show()

# Summary
print("\nFINAL RESULTS SUMMARY")
print(results_df.to_string(index=False))
best_model = results_df.iloc[0]
print(f"\nBest Model : {best_model['Model']}")
print(f"ROC-AUC    : {best_model['ROC-AUC']:.4f}")
print(f"Accuracy   : {best_model['Accuracy']:.4f}")