import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
from lightgbm import LGBMRegressor  # LightGBM kütüphanesini ekledik

# Protein özelliklerini hesaplama fonksiyonunu import et
from protein_features import calculate_protein_features, amino_acids

# Uyarı mesajlarını ekranda gösterme
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Başlangıç amino asit dizisi
# bu aminoasit dizisinin unipro id = E5BBQ3
initial_sequence = "ANPYERGPNPTDALLEARSGPFSVSEENVSRLSASGFGGGTIYYPRENNTYGAVAISPGYTGTEASIAWLGERIASHGFVVITIDTITTLDQPDSRAEQLNAALNHMINRASSTVRSRIDSSRLAVMGHSMGGGGSLRLASQRPDLKAAIPLTPWHLNKNWSSVTVPTLIIGADLDTIAPVATHAKPFYNSLPSSISKAYLELDGATHFAPNIPNKIIGKYSVAWLKRFVDNDTRYTQFLCPGPRDGLFGEVEEYRSTCPF"

# Substrat bölgesi (örnek olarak PETase enziminin aktif bölgesi)
# Bu değerler enzime göre değişebilir, gerekirse güncellenmeli
# Aşağıdaki değer E5BBQ3 nolu enzime aittir.
# substrat_pozisyonu_bul.py adlı dosya ile sorgulama yapılmıştır.
substrate_region = [100, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 171, 195, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214]
#substrate_region = [100, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 171, 195] # Amino asit pozisyonları (0'dan başlayarak)
#substrate_region = [100, 171, 195]

# 1. Mutasyon Fonksiyonu (Substrat bölgesi korumalı)
def mutate_sequence(sequence, substrate_region, mutation_rate=0.01):
    """Amino asit dizisinde rastgele mutasyonlar oluşturur (substrat bölgesi hariç)."""
    sequence = list(sequence)
    for i in range(len(sequence)):
        if i not in substrate_region and np.random.rand() < mutation_rate:
            sequence[i] = np.random.choice(list(amino_acids))
    return ''.join(sequence)

# 2. Veri Yükleme ve Model Eğitimi
print("Veri seti yükleniyor...")
df = pd.read_csv("enzim_data_tum_bilgiler.csv")
print(f"Toplam {len(df)} enzim verisi yüklendi.")

# Kategorik ve gereksiz sütunları çıkar
drop_columns = ['id', 'ec', 'uniprot_id', 'domain', 'organism', 'ogt','ogt_note', 'topt_note', 'sequence']

# Bağımsız değişkenler (X) ve hedef değişken (y) ayırma
X_full = df.drop(columns=['topt'] + drop_columns)
y = df['topt']  # Hedef değişken (topt)

print(f"Başlangıç özellik sayısı: {X_full.shape[1]}")

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LightGBM modeli eğitimi (RandomForest yerine)
print("LightGBM modeli eğitiliyor...")
lgbm_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,  # Uyarıları kapatır
    min_child_samples=5  # Daha kararlı dallanmalar için
)
lgbm_model.fit(X_train_scaled, y_train)

# Model performansını değerlendir
y_pred = lgbm_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"LightGBM Model Performansı - RMSE: {rmse:.2f}°C, R²: {r2:.4f}")

# Özellik isimleri (modelde kullanılan sırayla)
feature_names = X_full.columns.tolist()

# Helper function to convert feature dictionary to DataFrame with correct columns
def features_to_dataframe(features_dict):
    """
    Özellik sözlüğünü DataFrame'e dönüştürür, eksik özellikleri 0 olarak doldurur
    """
    # Tüm özellikleri içeren bir sözlük oluştur
    aligned_features = {}
    for feature in feature_names:
        aligned_features[feature] = features_dict.get(feature, 0.0)
    
    # Tek seferde DataFrame oluştur
    return pd.DataFrame([aligned_features])

# 3. Yönlendirilmiş Evrim Süreci
def directed_evolution(initial_sequence, substrate_region, target_topt=25.0, n_iterations=100, n_mutants=100, mutation_rate=0.01, convergence_threshold=0.1):
    """
    Yönlendirilmiş evrim algoritması ile Topt optimizasyonu (substrat bölgesi korumalı)
    """
    print("\n=== Yönlendirilmiş Evrim Başlıyor ===")
    print(f"Substrat bölgesi korunuyor: {substrate_region}")
    
    # İlk diziyi değerlendir
    start_time = time.time()
    initial_features = calculate_protein_features(initial_sequence)
    
    if initial_features is None:
        print("Başlangıç dizisi için özellik hesaplaması başarısız oldu.")
        return None, None, []
    
    # Özellikleri DataFrame'e dönüştür
    initial_features_df = features_to_dataframe(initial_features)
    
    # Ölçeklendirme ve Topt tahmini
    initial_features_scaled = scaler.transform(initial_features_df)
    initial_topt = lgbm_model.predict(initial_features_scaled)[0]  # LightGBM kullanıyoruz
    
    best_sequence = initial_sequence
    best_topt = initial_topt
    best_features = initial_features
    
    print(f"Başlangıç dizisi - Tahmini Topt: {best_topt:.2f}°C")
    print(f"Hedef Topt: {target_topt:.2f}°C")
    
    # Topt değerlerini kaydetmek için liste
    topt_history = [best_topt]
    iteration_times = []
    
    # Hedeften uzaklığı hesaplama fonksiyonu (fitness)
    def fitness(topt):
        return -abs(topt - target_topt) # Negatif uzaklık (0'a yakın değerler daha iyi)
    
    best_fitness = fitness(best_topt)
    
    # Yönlendirilmiş evrim süreci
    for iteration in range(n_iterations):
        # Hedeften sapma kontrolü - erken durdurma koşulu
        current_deviation = abs(best_topt - target_topt)
        if current_deviation <= 0.1:  # Hedeften sapma 0.1°C veya daha azsa
            print(f"\nHedef Topt değerine ulaşıldı! (Sapma: {current_deviation:.2f}°C)")
            print(f"İterasyon {iteration + 1}/{n_iterations}'da erken durduruluyor.")
            break
            
        iter_start_time = time.time()
        print(f"\nİterasyon {iteration + 1}/{n_iterations} başladı...")
        
        # Yeni mutantlar oluştur (substrat bölgesi korumalı)
        mutants = [mutate_sequence(best_sequence, substrate_region, mutation_rate) for _ in range(n_mutants)]
        
        # Yeni mutantların özelliklerini hesapla ve Topt değerlerini tahmin et
        mutant_topt_values = []
        
        for i, mutant in enumerate(mutants):
            # Özellikleri hesapla
            mutant_features = calculate_protein_features(mutant)
            
            if mutant_features is None:
                continue  # Bu mutantı atla
            
            # Özellikleri DataFrame'e dönüştür
            mutant_features_df = features_to_dataframe(mutant_features)
            
            # Ölçeklendirme ve Topt tahmini
            mutant_features_scaled = scaler.transform(mutant_features_df)
            mutant_topt = lgbm_model.predict(mutant_features_scaled)[0]  # LightGBM kullanıyoruz
            
            mutant_fitness = fitness(mutant_topt)
            
            mutant_topt_values.append((mutant, mutant_topt, mutant_fitness, mutant_features))
        
        # Mutantları fitness değerine göre sırala (en iyisi en üstte)
        mutant_topt_values.sort(key=lambda x: x[2], reverse=True)
        
        if mutant_topt_values:  # Eğer geçerli mutant varsa
            # En iyi mutantı seç
            best_mutant, best_mutant_topt, best_mutant_fitness, best_mutant_features = mutant_topt_values[0]
            
            # Eğer yeni mutant daha iyiyse, en iyi diziyi güncelle
            if best_mutant_fitness > best_fitness:
                improvement = abs(best_mutant_topt - best_topt)
                best_sequence = best_mutant
                best_topt = best_mutant_topt
                best_fitness = best_mutant_fitness
                best_features = best_mutant_features
                
                print(f"İyileşme! Yeni Topt: {best_topt:.2f}°C (Değişim: {improvement:.2f}°C)")
                
                # Eğer değişim çok küçükse, yakınsama kontrolü
                if improvement < convergence_threshold:
                    print(f"Yakınsama tespit edildi: Değişim ({improvement:.4f}°C) < Eşik ({convergence_threshold}°C)")
                    if abs(best_topt - target_topt) < 1.0:  # Hedeften 1°C'den az sapma varsa dur
                        print(f"Hedef Topt değerine yakın bir sonuca ulaşıldı!")
                        break
            else:
                print(f"İyileşme yok. En iyi Topt hala: {best_topt:.2f}°C")
        
        # Topt değerlerini kaydet
        topt_history.append(best_topt)
        
        # İterasyon süresini kaydet
        iter_time = time.time() - iter_start_time
        iteration_times.append(iter_time)
        
        print(f"İterasyon süresi: {iter_time:.2f} saniye")
        print(f"Hedeften sapma: {abs(best_topt - target_topt):.2f}°C")
    
    total_time = time.time() - start_time
    print(f"\nEvrim süreci tamamlandı. Toplam süre: {total_time:.2f} saniye")
    
    return best_sequence, best_topt, topt_history

# 4. Yönlendirilmiş Evrim'i Çalıştır
target_topt = 24.0  # Hedef Topt değeri tanımı

best_sequence, best_topt, topt_history = directed_evolution(
    initial_sequence=initial_sequence,
    substrate_region=substrate_region,  # Substrat bölgesi parametresi eklendi
    target_topt=target_topt,
    n_iterations=50,
    n_mutants=150,
    mutation_rate=0.08,
    convergence_threshold=0.01
)

# 5. Sonuçları Görselleştir
if topt_history:
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(topt_history)), topt_history, marker='o', linestyle='-', color='b')
    plt.axhline(y=target_topt, color='r', linestyle='--', label=f"Hedef Topt ({target_topt:.1f}°C)")
    plt.title("Yönlendirilmiş Evrim Sürecinde Topt Değerinin Değişimi (LightGBM)")
    plt.xlabel("İterasyon")
    plt.ylabel("Topt Değeri (°C)")
    plt.grid(True)
    plt.legend()
    plt.savefig('yonlendirilmis_evrim_grafik_lightgbm.png')  # LightGBM için farklı bir dosya adı
    plt.show()

# 6. Son Sonuçları Göster
if best_sequence and best_topt:
    print("\n=== Yönlendirilmiş Evrim Sonuçları (LightGBM) ===")  # LightGBM olduğunu belirttik
    print(f"Başlangıç Topt Değeri: {topt_history[0]:.2f}°C")
    print(f"Son Topt Değeri: {best_topt:.2f}°C")
    print(f"Toplam İyileşme: {abs(best_topt - topt_history[0]):.2f}°C")
    print(f"Hedeften Sapma: {abs(best_topt - target_topt):.2f}°C")
    print("\nEn İyi Amino Asit Dizisi:")
    
    # Uzun diziyi 80 karakterlik satırlara böl
    for i in range(0, len(best_sequence), 80):
        print(best_sequence[i:i+80])
    
    # Substrat bölgesi için kontrol
    print("\nSubstrat Bölgesi Amino Asitleri:")
    for pos in substrate_region:
        if pos < len(best_sequence):
            print(f"Pozisyon {pos}: {best_sequence[pos]}")
    
    # Sonuçları dosyaya kaydet
    with open('en_iyi_enzim_lightgbm.txt', 'w') as f:  # LightGBM için farklı bir dosya adı
        f.write(f"Topt Değeri: {best_topt:.2f}°C\n\n")
        f.write("Amino Asit Dizisi:\n")
        for i in range(0, len(best_sequence), 80):
            f.write(f"{best_sequence[i:i+80]}\n")
        f.write("\nSubstrat Bölgesi Amino Asitleri:\n")
        for pos in substrate_region:
            if pos < len(best_sequence):
                f.write(f"Pozisyon {pos}: {best_sequence[pos]}\n")
    
    print("\nSonuçlar 'en_iyi_enzim_lightgbm.txt' dosyasına kaydedildi.")
else:
    print("Evrim süreci başarısız oldu.")
