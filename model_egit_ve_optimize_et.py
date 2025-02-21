import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Veri Setini Yükle
df = pd.read_csv("enzyme_dataset_with_sequences_2643_samples.csv")
df = df.iloc[:300]

# 2. Veri Setini Hazırla
# Bağımsız değişkenler (X) ve hedef değişken (y) ayırma
X = df.drop(columns=['Topt', 'Amino_Asit_Dizisi'])  # Özellikler
y = df['Topt']  # Hedef değişken (Topt)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#arka arkara run edildiğinde aynı rastgele test setini oluşturması için

# Veriyi ölçeklendirme (StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Model Oluşturma ve Tahmin
# 3.1 Doğrusal Regresyon (Lasso ile Düzenlenmiş)
lasso_model = Lasso(alpha=0.01, max_iter=10000)  # Lasso Regression
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# 3.2 Random Forest Regresyon (Optimizasyon yok)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 3.3 Lojistik Regresyon (Topt < 25°C sınıflandırması için)
# Topt değerlerini kategorik hale getirme (Topt < 25°C ise 1, değilse 0)
y_train_class = (y_train < 25).astype(int)
y_test_class = (y_test < 25).astype(int)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train_class)
y_pred_logistic = logistic_model.predict(X_test)

# 4. Model Performansını Değerlendirme
# 4.1 Lasso Regression Performansı
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"Lasso Regression - MSE: {mse_lasso:.2f}, R²: {r2_lasso:.2f}")

# 4.2 Random Forest Regresyon Performansı
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")

# 4.3 Lojistik Regresyon Performansı
accuracy_logistic = accuracy_score(y_test_class, y_pred_logistic)
print(f"Lojistik Regresyon - Doğruluk: {accuracy_logistic:.2f}")

# 5. Özellik Ağırlıklarını İnceleme (Lasso için)
# Sıfır olmayan katsayıların sayısı
non_zero_coeffs = np.sum(lasso_model.coef_ != 0)
print(f"Lasso Regression - Sıfır Olmayan Katsayıların Sayısı: {non_zero_coeffs}")

# En önemli 10 özellik
feature_importance = pd.DataFrame({
    'Özellik': X.columns,
    'Katsayı': lasso_model.coef_
})
print("Lasso Regression - En Önemli 10 Özellik:")
print(feature_importance.sort_values(by='Katsayı', key=abs, ascending=False).head(10))

results_df = pd.DataFrame({
    'Gerçek Topt': y_test,
    'Lasso Tahmini': y_pred_lasso,
    'Random Forest Tahmini': y_pred_rf,
    'Lojistik Sınıflandırma (Topt < 25°C)': y_pred_logistic
})

# İlk 10 örneği göster
print("Tahmin Sonuçları:")
print(results_df.head(10))

# Tabloyu CSV'ye kaydet
results_df.to_csv("tahmin_sonuclari.csv", index=False)