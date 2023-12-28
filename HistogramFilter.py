import os
import cv2

#input_directory = 'Alzheimer_s DatasetHistogramEqualization/test_hist_equalized'
input_directory = 'Alzheimer_s DatasetHistogramEqualization/train_hist_equalized'
#input_directory = 'Alzheimer_s DatasetHistogramEqualization/val_hist_equalized'

# Median filtresi parametreleri
filter_size = 7

# Verilen bir dizindeki tüm görüntülere median filtresi uygulayan fonksiyon
def apply_histogram_filter(input_path):
    for image_file in os.listdir(input_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(input_path, image_file)
            img = cv2.imread(image_path)
            img_equalized = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                
                # Eşitlenmiş görüntüyü kaydetme
            
            cv2.imwrite(image_path, img_equalized)
# Filtreleme işlemi ve mevcut dosyaların üzerinde kaydetme
for directory in ['MildDemented_hist_equalized', 'ModerateDemented_hist_equalized', 'NonDemented_hist_equalized','VeryMildDemented_hist_equalized']:
    input_path = os.path.join(input_directory, directory)
    
    apply_histogram_filter(input_path)
