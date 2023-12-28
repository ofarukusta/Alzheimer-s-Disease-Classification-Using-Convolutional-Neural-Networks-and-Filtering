import os
import cv2

#input_directory = 'Alzheimer_s DatasetMedianFilter/train_median'
#input_directory = 'Alzheimer_s DatasetMedianFilter/test_median'
input_directory = 'Alzheimer_s DatasetMedianFilter/val_median'

# Median filtresi parametreleri
filter_size = 7

# Verilen bir dizindeki tüm görüntülere median filtresi uygulayan fonksiyon
def apply_median_filter(input_path):
    for image_file in os.listdir(input_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(input_path, image_file)
            img = cv2.imread(image_path)
            
            # Median filtresi uygulama
            median = cv2.medianBlur(img, filter_size)
            
            # Filtrelenmiş görüntüyle orijinal görüntüyü değiştirme
            cv2.imwrite(image_path, median)

# Filtreleme işlemi ve mevcut dosyaların üzerinde kaydetme
for directory in ['MildDemented', 'ModerateDemented', 'NonDemented','VeryMildDemented']:
    input_path = os.path.join(input_directory, directory)
    
    apply_median_filter(input_path)
