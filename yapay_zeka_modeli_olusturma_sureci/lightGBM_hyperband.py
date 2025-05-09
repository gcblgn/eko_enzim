import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from scipy.stats import uniform, loguniform
import warnings
warnings.filterwarnings('ignore')  # Uyarıları bastır

# LightGBM için basitleştirilmiş değerlendirme fonksiyonu
def evaluate_config(params, X_train, y_train, X_val, y_val, n_estimators):
    # n_estimators'ı güncelle
    params['n_estimators'] = n_estimators
    
    # LightGBM modeli
    model = LGBMRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Modeli eğit
    model.fit(X_train, y_train)
    
    # Değerlendirme
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    return {'rmse': rmse, 'r2': r2, 'model': model, 'params': params}


def simple_hyperband(X_train, y_train, X_val, y_val, max_iter=1000, eta=3, num_configs=20):
    print("Basitleştirilmiş Hyperband başlıyor...")
    
    # Random konfigürasyonlar oluştur
    configs = []
    for i in range(num_configs):
        config = {
            'learning_rate': float(loguniform(0.01, 0.3).rvs()),
            'max_depth': int(np.random.randint(3, 10)),
            'num_leaves': int(np.random.randint(10, 100)),
            'min_child_samples': int(np.random.randint(5, 30)),
            'subsample': float(np.random.uniform(0.5, 0.99)),
            'colsample_bytree': float(np.random.uniform(0.5, 0.99))
        }
        configs.append(config)
    
    # İlk iterasyon - düşük iterasyon sayısıyla tüm konfigürasyonları dene
    first_round = []
    first_iter = int(max_iter / eta**2)
    print(f"İlk tur: {num_configs} konfigürasyon, {first_iter} ağaç")
    
    for i, config in enumerate(configs):
        result = evaluate_config(config, X_train, y_train, X_val, y_val, first_iter)
        first_round.append(result)
        print(f"Konfigürasyon {i+1}/{num_configs}, RMSE: {result['rmse']:.4f}, R²: {result['r2']:.4f}")
    
    # En iyi 1/eta konfigürasyonu seç
    first_round.sort(key=lambda x: x['rmse'])  # RMSE küçük olması daha iyi
    survived = first_round[:int(num_configs/eta)]
    
    # İkinci iterasyon - daha yüksek iterasyon sayısıyla kalan konfigürasyonları dene
    second_round = []
    second_iter = int(max_iter / eta)
    print(f"\nİkinci tur: {len(survived)} konfigürasyon, {second_iter} ağaç")
    
    for i, result in enumerate(survived):
        config = result['params']
        result = evaluate_config(config, X_train, y_train, X_val, y_val, second_iter)
        second_round.append(result)
        print(f"Konfigürasyon {i+1}/{len(survived)}, RMSE: {result['rmse']:.4f}, R²: {result['r2']:.4f}")
    
    # En iyi 1/eta konfigürasyonu seç
    second_round.sort(key=lambda x: x['rmse'])
    finalists = second_round[:int(len(survived)/eta)]
    
    # Son iterasyon - tam iterasyon sayısıyla kalan konfigürasyonları dene
    final_round = []
    print(f"\nSon tur: {len(finalists)} konfigürasyon, {max_iter} ağaç")
    
    for i, result in enumerate(finalists):
        config = result['params']
        result = evaluate_config(config, X_train, y_train, X_val, y_val, max_iter)
        final_round.append(result)
        print(f"Konfigürasyon {i+1}/{len(finalists)}, RMSE: {result['rmse']:.4f}, R²: {result['r2']:.4f}")
    
    # En iyi konfigürasyonu döndür
    final_round.sort(key=lambda x: x['rmse'])
    best_config = final_round[0]
    
    print(f"\nEn iyi konfigürasyon: {best_config['params']}")
    print(f"En iyi RMSE: {best_config['rmse']:.4f}, R²: {best_config['r2']:.4f}")
    
    return best_config, first_round + second_round + final_round
    

try:
    # 1. Veri Setini Yükle
    print("Veri seti yükleniyor...")
    df = pd.read_csv("enzim_data_tum_bilgiler.csv")
    print(f"Toplam {len(df)} enzim verisi yüklendi.")
    
    # 2. Veri Setini Hazırla
    # Kategorik ve gereksiz sütunları çıkar
    drop_columns = ['id', 'ec', 'uniprot_id', 'domain', 'organism', 'ogt','ogt_note', 'topt_note', 'sequence']
    
    # Bağımsız değişkenler (X) ve hedef değişken (y) ayırma
    X_full = df.drop(columns=['topt'] + drop_columns)
    y = df['topt']  # Hedef değişken (topt)
    
    print(f"\nBaşlangıç özellik sayısı: {X_full.shape[1]}")
    
    # Veriyi eğitim, doğrulama ve test setlerine ayırma
    X_temp, X_test_full, y_temp, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
    X_train_full, X_val_full, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # Veriyi ölçeklendirme (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled_full = scaler.fit_transform(X_train_full)
    X_val_scaled_full = scaler.transform(X_val_full)
    X_test_scaled_full = scaler.transform(X_test_full)
    
    # 3. Özellik Seçimi için Ön Lasso Modeli
    print("\nÖzellik seçimi için Lasso modeli eğitiliyor...")
    feature_selection_start = time.time()
    from sklearn.linear_model import Lasso
    feature_selector = Lasso(alpha=0.1, max_iter=10000)
    feature_selector.fit(X_train_scaled_full, y_train)
    feature_selection_time = time.time() - feature_selection_start
    
    # Sıfır olmayan katsayılara sahip özelliklerin belirlenmesi
    selected_features_mask = feature_selector.coef_ != 0
    selected_features = X_full.columns[selected_features_mask]
    
    print(f"Lasso ile seçilen özellik sayısı: {len(selected_features)} / {len(X_full.columns)}")
    
    # 4. Seçilmiş özellikleri kullanarak veriyi hazırlama
    X_train = X_train_full[selected_features]
    X_val = X_val_full[selected_features]
    X_test = X_test_full[selected_features]
    
    # Seçilmiş özellikleri tekrar ölçeklendirme
    scaler_selected = StandardScaler()
    X_train_scaled = scaler_selected.fit_transform(X_train)
    X_val_scaled = scaler_selected.transform(X_val)
    X_test_scaled = scaler_selected.transform(X_test)
    
    # 5. Orjinal LightGBM Regresyon
    print("\nOrjinal LightGBM modeli eğitiliyor...")
    lgbm_start = time.time()
    lgbm_model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        min_child_samples=5
    )
    lgbm_model.fit(X_train_scaled, y_train)
    y_pred_lgbm = lgbm_model.predict(X_test_scaled)
    lgbm_time = time.time() - lgbm_start
    print("Orjinal LightGBM modeli eğitildi.")
    
    # 6. Basitleştirilmiş Hyperband Optimizasyonu
    print("\nHyperband optimizasyonu başlıyor...")
    hyperband_start = time.time()
    
    best_hyperband, all_results = simple_hyperband(
        X_train_scaled, y_train, 
        X_val_scaled, y_val,
        max_iter=300, eta=3, num_configs=9
    )
    
    hyperband_time = time.time() - hyperband_start
    print(f"Hyperband tamamlandı - Süre: {hyperband_time:.2f}s")
    
    # 7. En iyi modeli tüm eğitim + validation verisiyle yeniden eğit
    print("\nEn iyi model ile tam veri üzerinde eğitim yapılıyor...")
    
    # Eğitim ve doğrulama verilerini birleştir
    X_train_val = np.vstack([X_train_scaled, X_val_scaled])
    y_train_val = pd.concat([y_train, y_val])
    
    # En iyi modeli oluştur ve eğit
    best_model = LGBMRegressor(
        **best_hyperband['params'],
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    best_model.fit(X_train_val, y_train_val)
    
    # Test seti üzerinde tahmin
    y_pred_best = best_model.predict(X_test_scaled)
    
    # 8. Metrikleri hesapla
    # Orjinal model metrikleri
    mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
    rmse_lgbm = np.sqrt(mse_lgbm)
    r2_lgbm = r2_score(y_test, y_pred_lgbm)
    
    # Hyperband model metrikleri
    mse_best = mean_squared_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mse_best)
    r2_best = r2_score(y_test, y_pred_best)
    
    # 9. Tahmin Sonuçlarını Kaydet
    results_df = pd.DataFrame({
        'Gerçek Topt': y_test,
        'LightGBM Tahmini': y_pred_lgbm,
        'Hyperband LightGBM Tahmini': y_pred_best
    })
    
    # Hata hesapları
    results_df['LightGBM Hata'] = np.abs(results_df['Gerçek Topt'] - results_df['LightGBM Tahmini'])
    results_df['Hyperband LightGBM Hata'] = np.abs(results_df['Gerçek Topt'] - results_df['Hyperband LightGBM Tahmini'])
    
    # İlk 10 örneği göster
    print("\nTahmin Sonuçları (İlk 10 Örnek):")
    print(results_df.head(10))
    
    # 10. Sonuçları CSV'ye Kaydet
    output_path = "topt_tahmin_sonuclari_lightgbm_hyperband.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Tahmin sonuçları '{output_path}' dosyasına kaydedildi.")
    # Doğruluk ve AUC grafiklerini eklemek için aşağıdaki kod bloğunu 
        # mevcut kodun 11. özellik önem dereceleri bölümünden önce yerleştirebilirsiniz

    # Sınıflandırma metriklerini hesapla (doğruluk ve AUC)
    from sklearn.metrics import accuracy_score, roc_auc_score

    # Sıcaklık eşiği üzerinde ikili sınıflandırma
    TEMPERATURE_THRESHOLD = 40  # Bu değer mevcut değişkeni kullanır
    y_test_class = (y_test >= TEMPERATURE_THRESHOLD).astype(int)

    # Orijinal model için sınıflandırma tahminleri
    y_pred_lgbm_class = (y_pred_lgbm >= TEMPERATURE_THRESHOLD).astype(int)
    accuracy_lgbm = accuracy_score(y_test_class, y_pred_lgbm_class)
    # AUC için continuous değerleri olasılık gibi kullanabiliriz
    # Önce [0,1] aralığına normalize et
    y_prob_lgbm = (y_pred_lgbm - y_pred_lgbm.min()) / (y_pred_lgbm.max() - y_pred_lgbm.min())
    auc_lgbm = roc_auc_score(y_test_class, y_prob_lgbm)

    # Hyperband model için sınıflandırma tahminleri  
    y_pred_best_class = (y_pred_best >= TEMPERATURE_THRESHOLD).astype(int)
    accuracy_best = accuracy_score(y_test_class, y_pred_best_class)
    # AUC için continuous değerleri normalize et
    y_prob_best = (y_pred_best - y_pred_best.min()) / (y_pred_best.max() - y_pred_best.min())
    auc_best = roc_auc_score(y_test_class, y_prob_best)

    # Performans özetine sınıflandırma metriklerini ekle
    print("\n--- Orjinal LightGBM Sınıflandırma Performansı ---")
    print(f"Doğruluk: {accuracy_lgbm:.4f}, AUC: {auc_lgbm:.4f}")

    print("\n--- Hyperband LightGBM Sınıflandırma Performansı ---")
    print(f"Doğruluk: {accuracy_best:.4f}, AUC: {auc_best:.4f}")

    # Doğruluk ve AUC için ek görselleştirme
    plt.figure(figsize=(12, 5))

    # Doğruluk (Accuracy) ve AUC karşılaştırması
    plt.subplot(1, 2, 1)
    models = ['Orijinal LightGBM', 'Hyperband LightGBM']
    accuracy_values = [accuracy_lgbm, accuracy_best]
    auc_values = [auc_lgbm, auc_best]

    x = np.arange(len(models))
    width = 0.35

    bars1 = plt.bar(x - width/2, accuracy_values, width, label='Doğruluk', color='tab:blue')
    bars2 = plt.bar(x + width/2, auc_values, width, label='AUC', color='tab:orange')

    plt.ylabel('Değer')
    plt.title('Doğruluk ve AUC Karşılaştırması')
    plt.xticks(x, models)
    plt.ylim(0, 1.0)
    plt.legend()

    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')

    # RMSE ve R² karşılaştırması
    plt.subplot(1, 2, 2)
    rmse_values = [rmse_lgbm, rmse_best]
    r2_values = [r2_lgbm, r2_best]

    # RMSE için ayrı ölçek (sol y ekseni)
    ax1 = plt.gca()
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE ve R² Karşılaştırması')
    bars_rmse = ax1.bar(x - width/2, rmse_values, width, label='RMSE', color='crimson')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)

    # RMSE değerlerini çubukların üzerine yazdır
    for i, v in enumerate(rmse_values):
        ax1.text(i - width/2, v + 0.5, f'{v:.2f}', ha='center', va='bottom')

    # R² için ayrı ölçek (sağ y ekseni)
    ax2 = ax1.twinx()
    ax2.set_ylabel('R²')
    bars_r2 = ax2.bar(x + width/2, r2_values, width, label='R²', color='royalblue')

    # R² değerlerini çubukların üzerine yazdır
    for i, v in enumerate(r2_values):
        ax2.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    # Her iki ölçek için ayrı legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('lightgbm_hyperband_tum_metrikler.png', dpi=300)
    plt.close()

    print("Tüm metrik karşılaştırmaları 'lightgbm_hyperband_tum_metrikler.png' dosyasına kaydedildi.")
    
    # 11. Özellik önem derecelerini kaydet
    features_df = pd.DataFrame({
        'Özellik': selected_features,
        'Orijinal Önem': lgbm_model.feature_importances_,
        'Hyperband Önem': best_model.feature_importances_
    }).sort_values(by='Hyperband Önem', ascending=False)
    
    features_path = "lightgbm_hyperband_ozellik_onemleri.csv"
    features_df.to_csv(features_path, index=False)
    print(f"Özellik önem dereceleri '{features_path}' dosyasına kaydedildi.")
    
    # 12. Performans özeti
    print("\n=== Performans Özeti ===")
    print(f"Toplam özellik sayısı: {X_full.shape[1]}")
    print(f"Seçilen özellik sayısı: {len(selected_features)} ({len(selected_features)/X_full.shape[1]*100:.1f}%)")
    
    print("\n--- Orjinal LightGBM Model Performansı ---")
    print(f"RMSE: {rmse_lgbm:.2f}°C, R²: {r2_lgbm:.4f}, Süre: {lgbm_time:.2f} saniye")
    
    print("\n--- Hyperband LightGBM Model Performansı ---")
    print(f"RMSE: {rmse_best:.2f}°C, R²: {r2_best:.4f}, Süre: {hyperband_time:.2f} saniye")
    
    print(f"\nÖzellik seçimi süresi: {feature_selection_time:.2f} saniye")
    print(f"Toplam süre: {feature_selection_time + lgbm_time + hyperband_time:.2f} saniye")
    
    # 13. Görselleştirme
    plt.figure(figsize=(12, 5))
    
    # RMSE karşılaştırması
    plt.subplot(1, 2, 1)
    models = ['Orijinal LightGBM', 'Hyperband LightGBM']
    rmse_values = [rmse_lgbm, rmse_best]
    
    plt.bar(models, rmse_values, color='crimson')
    plt.ylabel('RMSE (°C)')
    plt.title('RMSE Karşılaştırması')
    
    # Değerleri çubukların üzerine yazdır
    for i, v in enumerate(rmse_values):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    
    # R² karşılaştırması
    plt.subplot(1, 2, 2)
    r2_values = [r2_lgbm, r2_best]
    
    plt.bar(models, r2_values, color='royalblue')
    plt.ylabel('R²')
    plt.title('R² Karşılaştırması')
    
    # Değerleri çubukların üzerine yazdır
    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('lightgbm_hyperband_karsilastirma.png', dpi=300)
    plt.close()
    
    print("Karşılaştırma grafikleri 'lightgbm_hyperband_karsilastirma.png' dosyasına kaydedildi.")
    
    print("\nİşlem tamamlandı!")
    
except Exception as e:
    print(f"HATA: {e}")
    import traceback
    traceback.print_exc()