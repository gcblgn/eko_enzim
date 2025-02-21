import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from modlamp.descriptors import PeptideDescriptor
from modlamp.descriptors import GlobalDescriptor

# 1. Veri Setini Yükle
df = pd.read_csv("enzyme_dataset_with_sequences_2643_samples.csv")
#df = df.iloc[:1000]  # İlk 100 örneği kullan

def calculate_features(sequence):
    """Amino asit dizisinden özellikleri hesaplar (modLAMP kullanarak)."""
    descriptor = GlobalDescriptor(sequence)
    descriptor.calculate_all()
    return descriptor.descriptor[0]

# 2. Veri Setini Hazırla
# Önce tüm amino asit dizileri için özellikleri hesapla
X_features = np.array([calculate_features(seq) for seq in df['Amino_Asit_Dizisi']])
y = df['Topt']  # Hedef değişken (Topt)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme (StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Random Forest Modelini Eğitme
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Yönlendirilmiş Evrim Süreci
def mutate_sequence(sequence, substrate_region, mutation_rate=0.01):
    """Amino asit dizisinde rastgele mutasyonlar oluşturur (substrat bölgesi hariç)."""
    sequence = list(sequence)
    for i in range(len(sequence)):
        if i not in substrate_region and np.random.rand() < mutation_rate:
            sequence[i] = np.random.choice(amino_acids)
    return ''.join(sequence)

# Başlangıç amino asit dizisi (rastgele bir örnek)
amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
initial_sequence = df['Amino_Asit_Dizisi'].sample(1).values[0]

# Substrat bölgesi (örnek olarak PETase enziminin aktif bölgesi)
substrate_region = [205, 206, 237, 180, 185, 161]  # Amino asit pozisyonları (0'dan başlayarak)

# Yönlendirilmiş evrim parametreleri
n_iterations = 35  # Toplam iterasyon sayısı
n_mutants = 1000  # Her iterasyonda üretilecek mutant sayısı
mutation_rate = 0.01  # Çok düşük mutasyon oranı

# İlk diziyi değerlendir
initial_features = calculate_features(initial_sequence)
initial_features_scaled = scaler.transform([initial_features])
best_sequence = initial_sequence
best_topt = rf_model.predict(initial_features_scaled)[0]

# Topt değerlerini kaydetmek için liste
topt_values = [best_topt]

# Yönlendirilmiş evrim süreci
for iteration in range(n_iterations):
    print(f"Iterasyon {iteration + 1} başladı...")
    
    # Yeni mutantlar oluştur
    mutants = [mutate_sequence(best_sequence, substrate_region, mutation_rate) for _ in range(n_mutants)]
    
    # Yeni mutantların özelliklerini hesapla ve Topt değerlerini tahmin et
    mutants_features = [calculate_features(mutant) for mutant in mutants]
    mutants_features_scaled = scaler.transform(mutants_features)
    mutants_topt = rf_model.predict(mutants_features_scaled)
    
    # En iyi 50 mutantı seç ve rastgele birini ana enzim olarak al
    top_50_indices = np.argsort(mutants_topt)[:50]
    best_mutant_index = np.random.choice(top_50_indices)
    best_mutant_sequence = mutants[best_mutant_index]
    best_mutant_topt = mutants_topt[best_mutant_index]
    
    # Eğer yeni mutant daha iyiyse, en iyi diziyi güncelle
    if best_mutant_topt < best_topt:
        best_sequence = best_mutant_sequence
        best_topt = best_mutant_topt
    
    # Topt değerlerini kaydet
    topt_values.append(best_topt)
    
    # İlerlemeyi yazdır
    print(f"Iterasyon {iteration + 1}: En İyi Topt = {best_topt:.2f}°C")

# 5. Topt Değerlerinin Değişim Grafiğini Çiz
plt.figure(figsize=(10, 6))
plt.plot(range(n_iterations + 1), topt_values, marker='o', linestyle='-', color='b')
plt.title("Yönlendirilmiş Evrim Sürecinde Topt Değerinin Değişimi")
plt.xlabel("Iterasyon")
plt.ylabel("Topt Değeri (°C)")
plt.grid(True)
plt.axhline(y=25, color='r', linestyle='--', label="Hedef Topt (25°C)")
plt.legend()
plt.show()

# 6. Sonuçları Yazdır
print(f"Sonuç: En İyi Topt Değeri = {best_topt:.2f}°C")
print(f"En İyi Amino Asit Dizisi: {best_sequence}")