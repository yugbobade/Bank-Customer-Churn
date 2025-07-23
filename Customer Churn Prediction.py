### 🚨 Intelligent Customer Churn Prediction System for Banking Sector

---

## 📁 1. Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

## 📄 2. Load Dataset
```python
df = pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Bank-Customer-Churn-Prediction/master/Churn_Modelling.csv")
df.head()
```

---

## 🧹 3. Data Preprocessing

### Drop unwanted columns
```python
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
```

### Encode Categorical Variables
```python
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df = pd.get_dummies(df, drop_first=True)  # For Geography
```

---

## ⚖️ 4. Feature Scaling
```python
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Exited', axis=1))
X = pd.DataFrame(scaled_features, columns=df.drop('Exited', axis=1).columns)
y = df['Exited']
```

---

## ✂️ 5. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🌲 6. Train Model: Random Forest
```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 📊 7. Evaluate the Model
```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

---

## 🌟 8. Feature Importance
```python
importances = model.feature_importances_
feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
feat_importance.plot(kind='bar', figsize=(10, 6))
plt.title("Feature Importance")
plt.show()
```

---

## 🔍 9. SHAP Explainability
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)
```

---

## 🧠 10. Try Other Models (Simplified)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(),
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, pred):.2f}")
```

---

## 💾 11. Save Model
```python
import joblib
joblib.dump(model, "customer_churn_model.pkl")
```

---

## ✅ Summary:
- We built a **customer churn prediction system**
- Used a simple **Random Forest model**
- Added **feature importance** and **SHAP** for explainability
- Compared with other models
- **Saved the model** for production

🎯 Perfect to showcase in interviews!

---

Shall we now write the **README with emojis and clean sections** to make it ✨ look like an industry-grade GitHub project?
