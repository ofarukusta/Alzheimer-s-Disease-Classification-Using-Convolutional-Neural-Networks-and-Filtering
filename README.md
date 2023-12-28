# Alzheimer’s Disease Classification Using Deep Learning with Convolutional Neural Networks
 CNN ve çeşitli filtreleme araçları kullanılarak Yapay Zeka kontrollü Alzheimer Süreç Tespit Uygulaması

Geçen on yılda, yüksek hacimli biyomedikal veri setlerinin (nörogörüntüleme ve ilgili biyolojik veriler) hızla artması, makine öğrenimi (ML) alanındaki gelişmelerle eş zamanlı olarak, nörodejeneratif ve nöropsikiyatrik bozuklukların teşhisi ve prognozu için yeni olanaklar açmıştır. Örnek araştırmalarda da görüldüğü üzere Makine Öğrenmesi bazlı yapıların sektörde çığır açan bir değişime öncülük ettiği bilinmektedir.

Daha önce gözlemlenen birçok çalışmada CNN kullanılmış olsa da filtre çıkışına CNN kullanılmış bir uygulamaya rastlanmamış olması sebebiyle, proje özgünlüğünü korumaktadır.
# Proje Akış Diyagramı
![image](https://github.com/ofarukusta/Alzheimer-s-Disease-Classification-Using-Convolutional-Neural-Networks-and-Filtering/assets/110857814/dc4fc811-b6d2-470f-962d-fc0757b4f8d7)

## Alçak Geçiren Filtre:
Alçak geçiren filtre, frekans domenindeki düşük frekanslı bileşenleri korurken yüksek frekanslı bileşenleri zayıflatır. Genellikle, görüntülerdeki gürültüyü azaltmak ve pürüzsüzleştirmek için kullanılır.
Görüntüdeki her bir piksel, komşu piksellerin ağırlıklı ortalaması ile değiştirilir.
Bu ağırlıklar, bir filtre çekirdeği (kernel) ile belirlenir. Çekirdek, genellikle aynı boyutlu bir matristir ve bu matrisin merkezi (genellikle köşegen) önemli ağırlıklar içerir.
Görüntü üzerinde kayan pencere (sliding window) şeklinde ilerlerken, her piksel için çekirdek uygulanır ve çıktı hesaplanır.

```
python lowPassFilter.py
```

## Yüksek Geçiren Filtre(Pretwitt):

Prewitt filtresi, kenarları belirlemek ve vurgulamak için kullanılan bir kenar tespitme filtresidir. Dikey ve yatay yönde kenarları tespit etmek için iki ayrı filtre çekirdeği kullanır. Tensorflow’un doğrudan YGF bağıntılı filtrasyon işlemi olmadığından YGF tabanlı Prewitt kullanılmıştır.
Prewitt filtresi, kenar tespitinde kullanılan matrislerden oluşur. Dikey kenarları tespit etmek için bir dikey türev hesabı yapar, yatay kenarları tespit etmek için ise yatay türev hesabı yapar.
Görüntü üzerinde her bir piksel için dikey ve yatay filtreler uygulanarak kenarlar belirlenir. Bu işlem, kenarların yoğunluğunu ve yönünü tespit etmeye yarar.

![image](https://github.com/ofarukusta/Alzheimer-s-Disease-Classification-Using-Convolutional-Neural-Networks-and-Filtering/assets/110857814/b268d207-2fa1-46db-85e4-a971c6f9f87d)

![image](https://github.com/ofarukusta/Alzheimer-s-Disease-Classification-Using-Convolutional-Neural-Networks-and-Filtering/assets/110857814/dd564511-46b5-40e2-8312-13d4ff2c63c6)

```
python highPassFilter.py
```

## Median Filtre:
Median filtresi, bir pikselin değerini, belirli bir pencerenin (örneğin, 3x3 veya 5x5 boyutlu bir pencerenin) medyanı ile değiştirir. Gürültüyü azaltmak için kullanılır ve özellikle salt ve impuls gibi gürültülerin giderilmesinde etkilidir.
Median filtresi, piksel değerlerini sıralar ve ortanca (median) değeri alır.
Görüntü üzerinde her bir piksel için, belirlenen pencere içindeki pikseller sıralanır ve ortanca değer pikselin yeni değeri olarak atanır.

![image](https://github.com/ofarukusta/Alzheimer-s-Disease-Classification-Using-Convolutional-Neural-Networks-and-Filtering/assets/110857814/b5b8d396-2d49-4074-89dd-fbe1affdc0db)

```
python MedianFilter.py
```

## Histogram Equalized
Histogram eşitleme, bir görüntünün kontrastını artırmak ve genellikle görüntünün daha iyi anlaşılabilir olmasını sağlamak için kullanılır. Görüntüdeki piksel yoğunluklarının dağılımını değiştirerek görüntünün histogramını genişletir.
Histogram eşitleme, görüntüdeki piksel yoğunluklarını değiştirerek görüntü histogramının eşitlenmesini sağlar.
*Öncelikle, görüntünün histogramı hesaplanır ve piksel değerlerine göre olasılıkları elde edilir.
*Ardından, kümülatif dağılım fonksiyonu oluşturulur. Bu, her bir piksel değerinin kümülatif olasılık yoğunluğunu hesaplar.
*Son adımda, bu kümülatif fonksiyon kullanılarak görüntüdeki piksel değerleri eşitlenir. Bu, daha geniş bir renk aralığına sahip daha kontrastlı bir görüntü elde etmeyi amaçlar.

![image](https://github.com/ofarukusta/Alzheimer-s-Disease-Classification-Using-Convolutional-Neural-Networks-and-Filtering/assets/110857814/72c7c70d-b6b3-44f2-a161-bb50b93c1b41)

```
python HistogramFilter.py
```

### Filtrelerin Görüntü Üzerinden Kıyaslanması:
![image](https://github.com/ofarukusta/Alzheimer-s-Disease-Classification-Using-Convolutional-Neural-Networks-and-Filtering/assets/110857814/bd5a9af5-db0f-437b-8051-1c028374b77d)

### Örnek olarak Alçak Geçiren Filtre Test Sonuçları:
![image](https://github.com/ofarukusta/Alzheimer-s-Disease-Classification-Using-Convolutional-Neural-Networks-and-Filtering/assets/110857814/38ef7504-f7b8-43a5-b4f5-cb4fa2bcee2a)
![image](https://github.com/ofarukusta/Alzheimer-s-Disease-Classification-Using-Convolutional-Neural-Networks-and-Filtering/assets/110857814/f3037c05-e9f5-4bf3-8f78-b059a1f3665e)


```
Python proje.py
```


