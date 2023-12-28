import matplotlib.pyplot as plt

# Beş farklı resmin yolları
image_paths = [
    {"path":'Filtre_Kontrol\\Original.jpg', "title":"Original Dataset"},
    {"path":'Filtre_Kontrol\\HighPass.jpg',"title":"HighPass Filter"},
    {"path":'Filtre_Kontrol\\Histogram.jpg', "title":"Histogram Equalization"},
    {"path":'Filtre_Kontrol\\LowPass.jpg', "title":"LowPass Filter"},
    {"path":'Filtre_Kontrol\\Median.jpg', "title":"Median Filter"}
]

# Plot oluşturma
plt.figure(figsize=(15, 5))  # Plotun boyutunu ayarlayabilirsiniz

# Beş resmi tek bir plot içinde gösterme
for i, item in enumerate(image_paths, 1):
    plt.subplot(1, 5, i)
    img = plt.imread(item['path'])
    plt.imshow(img)
    plt.axis('off')
    plt.title(item['title'])  # Her bir resmin başlığını atama

plt.tight_layout()
plt.show()