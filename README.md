# Makine Öğrenimi Destekli Yönlendirilmiş Evrim ile Deniz Plastiklerini Parçalayan Eko-Enzim Geliştirilmesi

## Özet  
Plastik kirliliği günümüzün en büyük sorunlarından biridir. Plastik çöpler, **deniz çöplerinin %80'ini** oluşturarak küresel bir tehdit haline gelmiştir. *Thermobifida fusca Cutinase* enzimi plastikleri parçalama yeteneğine sahiptir fakat çalışma sıcaklığı deniz yüzeyinden farklı olduğu için kullanılamamaktadır. Projenin amacı, *Thermobifida fusca Cutinase* enziminin optimum çalışma sıcaklığını deniz suyu sıcaklıklarına (20-25°C) ayarlamaktır.

![image](https://github.com/user-attachments/assets/fead1a4b-571b-4550-bea8-d1a4bfd7cc4e)

**Şekil 1:**
*Thermobifida fusca Cutinase*

 Proje iki aşamalı olarak gerçekleştirilmiştir: İlk aşamada, oluşturulacak varyantların optimum çalışma sıcaklığı değerini tahmin etmek için **BRENDA** veritabanından alınan **2676 enzim verisi** kullanılarak makine öğrenimi destekli tahmin modelleri eğitilmiştir. Lojistik Regresyon, Lassolu Doğrusal Regresyon, XGBoost, LightGBM ve Rastgele Orman algoritmaları karşılaştırılmış, optimum sıcaklık tahmininde en iyi performansı gösteren **LightGBM yöntemi** seçilmiştir ve model üzerinde **hiperparametre optimizasyonları** uygulanmıştır. İkinci aşamada, yönlendirilmiş evrim *(in silico)* uygulanarak enzimin mutasyonları tasarlanmış ve her iterasyonda düşük sıcaklıkta çalışan en başarılı mutant belirlenmiştir. Ardından enzimin gerçek doğada faaliyet gösterip gösteremeyeceği **Progen2 adlı LLM ve AlphaFold 2 adlı yapay zeka** ile test edilmiştir.
Ortalama 15 iterasyon sonunda, enzimin optimum sıcaklığı 60.31°C’den 24.005°C’ye düşürülmüş ve yeni bir **“Eko-Enzim”** tasarlanmıştır. 


### Anahtar Kelimeler  
**Deniz Plastik Kirliliği, Makine Öğrenimi, Enzim Mühendisliği, Yönlendirilmiş Evrim.**

**Hazırlayan: Gökçe Ceyda Bilgin /**
**Bursa Halil İnalcık Bilim ve Sanat Merkezi**
