import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
# XGBoost ve LightGBM kütüphaneleri
import xgboost as xgb
from lightgbm import LGBMRegressor


# Bu değeri değiştirerek farklı sıcaklık eşikleri için sınıflandırma yapılabilir
TEMPERATURE_THRESHOLD = 40

# 1. Veri Setini Yükle
print("Veri seti yükleniyor...")
df = pd.read_csv("enzim_data_tum_bilgiler.csv")
print(f"Toplam {len(df)} enzim verisi yüklendi.")

# Veri seti hakkında genel bilgi
print("\nVeri Seti Özeti:")
print(f"Sütun sayısı: {len(df.columns)}")
print(f"Eksik değer içeren satır sayısı: {df.isnull().any(axis=1).sum()}")

# 2. Veri Setini Hazırla
# Kategorik ve gereksiz sütunları çıkar
drop_columns = ['id', 'ec', 'uniprot_id', 'domain', 'organism', 'ogt','ogt_note', 'topt_note', 'sequence']

# Bağımsız değişkenler (X) ve hedef değişken (y) ayırma
X_full = df.drop(columns=['topt'] + drop_columns)
y = df['topt']  # Hedef değişken (topt)

print(f"\nBaşlangıç özellik sayısı: {X_full.shape[1]}")

# Veriyi eğitim ve test setlerine ayırma
X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme (StandardScaler)
scaler = StandardScaler()
X_train_scaled_full = scaler.fit_transform(X_train_full)
X_test_scaled_full = scaler.transform(X_test_full)

# 3. Özellik Seçimi için Ön Lasso Modeli
print("\nÖzellik seçimi için Lasso modeli eğitiliyor...")
# Süresi ölç
feature_selection_start = time.time()
feature_selector = Lasso(alpha=0.1, max_iter=10000)
feature_selector.fit(X_train_scaled_full, y_train)
feature_selection_time = time.time() - feature_selection_start

# Sıfır olmayan katsayılara sahip özelliklerin belirlenmesi
selected_features_mask = feature_selector.coef_ != 0
selected_features = X_full.columns[selected_features_mask]

print(f"Lasso ile seçilen özellik sayısı: {len(selected_features)} / {len(X_full.columns)}")

# 4. Seçilmiş özellikleri kullanarak veriyi hazırlama
X_train = X_train_full[selected_features]
X_test = X_test_full[selected_features]

# Seçilmiş özellikleri tekrar ölçeklendirme
scaler_selected = StandardScaler()
X_train_scaled = scaler_selected.fit_transform(X_train)
X_test_scaled = scaler_selected.transform(X_test)

# 5. Seçilmiş özelliklerle model eğitimi
print("\nSeçilen özelliklerle modeller eğitiliyor...")

# Çalışma sürelerini saklayacak değişkenler
lasso_time = 0
rf_time = 0
logistic_time = 0
xgb_time = 0
lgbm_time = 0

# 5.1 Lasso Regresyon
lasso_start = time.time()
lasso_model = Lasso(alpha=0.1, max_iter=10000)  
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
lasso_time = time.time() - lasso_start
print("Lasso modeli eğitildi.")

# 5.2 Random Forest Regresyon
rf_start = time.time()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_time = time.time() - rf_start
print("Random Forest modeli eğitildi.")

# 5.3 XGBoost Regresyon
xgb_start = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
xgb_time = time.time() - xgb_start
print("XGBoost modeli eğitildi.")

# 5.4 LightGBM Regresyon
lgbm_start = time.time()
lgbm_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,  # Bu satırı ekleyin - uyarıları kapatır
    min_child_samples=5  # Bu satırı da ekleyin - daha kararlı dallanmalar için
)
lgbm_model.fit(X_train_scaled, y_train)
y_pred_lgbm = lgbm_model.predict(X_test_scaled)
lgbm_time = time.time() - lgbm_start
print("LightGBM modeli eğitildi.")

# 5.5 Lojistik Regresyon (Sınıflandırma)
logistic_start = time.time()
y_train_class = (y_train >= TEMPERATURE_THRESHOLD).astype(int)
y_test_class = (y_test >= TEMPERATURE_THRESHOLD).astype(int)

logistic_model = LogisticRegression(C=1e-10, max_iter=5000, solver='sag')
logistic_model.fit(X_train_scaled, y_train_class)
y_pred_logistic = logistic_model.predict(X_test_scaled)
logistic_time = time.time() - logistic_start
print("Lojistik Regresyon modeli eğitildi.")

# 6. Metrikleri hesapla (ama şimdi gösterme)
# Regresyon metrikleri
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mse_lgbm)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

# Sınıflandırma metriği
accuracy_logistic = accuracy_score(y_test_class, y_pred_logistic)

# 7. Tahmin Sonuçlarını Kaydet
results_df = pd.DataFrame({
    'Gerçek Topt': y_test,
    'Lasso Tahmini': y_pred_lasso,
    'Random Forest Tahmini': y_pred_rf,
    'XGBoost Tahmini': y_pred_xgb,
    'LightGBM Tahmini': y_pred_lgbm,
    f'Termofilik Sınıflandırma (>= {TEMPERATURE_THRESHOLD}°C)': y_pred_logistic
})

# Hata hesapları
results_df['Lasso Hata'] = np.abs(results_df['Gerçek Topt'] - results_df['Lasso Tahmini'])
results_df['RF Hata'] = np.abs(results_df['Gerçek Topt'] - results_df['Random Forest Tahmini'])
results_df['XGBoost Hata'] = np.abs(results_df['Gerçek Topt'] - results_df['XGBoost Tahmini'])
results_df['LightGBM Hata'] = np.abs(results_df['Gerçek Topt'] - results_df['LightGBM Tahmini'])

# İlk 10 örneği göster
cols_to_show = ['Gerçek Topt', 'Lasso Tahmini', 'Random Forest Tahmini', 
                'XGBoost Tahmini', 'LightGBM Tahmini',
                f'Termofilik Sınıflandırma (>= {TEMPERATURE_THRESHOLD}°C)']
print("\nTahmin Sonuçları (İlk 10 Örnek):")
print(results_df[cols_to_show].head(10))

# 8. Sonuçları CSV'ye Kaydet
output_path = "topt_tahmin_sonuclari_all_models.csv"
results_df.to_csv(output_path, index=False)
print(f"Tahmin sonuçları '{output_path}' dosyasına kaydedildi.")

# 9. Seçilen özellikleri kaydet
selected_features_df = pd.DataFrame({
    'Özellik': selected_features
})
features_path = "secilen_ozellikler.csv"
selected_features_df.to_csv(features_path, index=False)
print(f"Seçilen {len(selected_features)} özellik '{features_path}' dosyasına kaydedildi.")

# 10. En iyi modeli belirle
models_rmse = {
    'Lasso': rmse_lasso,
    'Random Forest': rmse_rf,
    'XGBoost': rmse_xgb,
    'LightGBM': rmse_lgbm
}
best_model = min(models_rmse, key=models_rmse.get)

models_r2 = {
    'Lasso': r2_lasso,
    'Random Forest': r2_rf,
    'XGBoost': r2_xgb,
    'LightGBM': r2_lgbm
}
best_r2_model = max(models_r2, key=models_r2.get)

# 11. Özellik önem derecelerini göster (en iyi model için)
if best_model == 'Random Forest':
    feature_importance = pd.DataFrame({
        'Özellik': selected_features,
        'Önem': rf_model.feature_importances_
    }).sort_values(by='Önem', ascending=False)
elif best_model == 'XGBoost':
    feature_importance = pd.DataFrame({
        'Özellik': selected_features,
        'Önem': xgb_model.feature_importances_
    }).sort_values(by='Önem', ascending=False)
elif best_model == 'LightGBM':
    feature_importance = pd.DataFrame({
        'Özellik': selected_features,
        'Önem': lgbm_model.feature_importances_
    }).sort_values(by='Önem', ascending=False)
    
if best_model in ['Random Forest', 'XGBoost', 'LightGBM']:
    feature_importance_path = f"{best_model.lower().replace(' ', '_')}_ozellik_onem_dereceleri.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"\n{best_model} için özellik önem dereceleri '{feature_importance_path}' dosyasına kaydedildi.")

# 12. Performans özeti 
print("\n=== Performans Özeti ===")
print(f"Toplam özellik sayısı: {X_full.shape[1]}")
print(f"Seçilen özellik sayısı: {len(selected_features)} ({len(selected_features)/X_full.shape[1]*100:.1f}%)")
print("\n--- Model Performansları ---")
print(f"Lasso - RMSE: {rmse_lasso:.2f}°C, R²: {r2_lasso:.4f}, Süre: {lasso_time:.2f} saniye")
print(f"Random Forest - RMSE: {rmse_rf:.2f}°C, R²: {r2_rf:.4f}, Süre: {rf_time:.2f} saniye")
print(f"XGBoost - RMSE: {rmse_xgb:.2f}°C, R²: {r2_xgb:.4f}, Süre: {xgb_time:.2f} saniye")
print(f"LightGBM - RMSE: {rmse_lgbm:.2f}°C, R²: {r2_lgbm:.4f}, Süre: {lgbm_time:.2f} saniye")
print(f"Lojistik Regresyon - Doğruluk: {accuracy_logistic:.4f}, Süre: {logistic_time:.2f} saniye")

print(f"\nÖzellik seçimi süresi: {feature_selection_time:.2f} saniye")
print(f"Toplam model eğitim süresi: {lasso_time + rf_time + xgb_time + lgbm_time + logistic_time:.2f} saniye")

print(f"\n=== En İyi Model: {best_model} ===")
print(f"En Düşük RMSE: {models_rmse[best_model]:.2f}°C")
print(f"En Yüksek R²: {models_r2[best_r2_model]:.4f} ({best_r2_model} modeli)")

# 13. En yüksek hata yapılan örnekleri göster (en iyi model için)
if best_model == 'Lasso':
    error_column = 'Lasso Hata'
    pred_column = 'Lasso Tahmini'
elif best_model == 'Random Forest':
    error_column = 'RF Hata'
    pred_column = 'Random Forest Tahmini'
elif best_model == 'XGBoost':
    error_column = 'XGBoost Hata'
    pred_column = 'XGBoost Tahmini'
elif best_model == 'LightGBM':
    error_column = 'LightGBM Hata'
    pred_column = 'LightGBM Tahmini'
