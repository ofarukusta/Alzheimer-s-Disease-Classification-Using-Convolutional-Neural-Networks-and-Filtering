import os
import tensorflow as tf
import numpy as np

# Ana dizin ve düşük geçişli filtre uygulanacak klasörlerin yolları
input_directory = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s Dataset'
output_directory = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetLowPass'

# Gaussian filtresi parametreleri
filter_shape = (5, 5)
sigma = 1.0

# Verilen bir dizindeki görüntülere düşük geçişli filtre uygulayan fonksiyon
def apply_low_pass_filter(input_path, output_path):
    for class_folder in os.listdir(input_path):
        class_path = os.path.join(input_path, class_folder)
        output_class_path = os.path.join(output_path, class_folder + '_lowpass')
        
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
        
        for image_file in os.listdir(class_path):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(class_path, image_file)
                img = tf.io.read_file(image_path)
                img = tf.image.decode_image(img, channels=3)
                img = tf.image.convert_image_dtype(img, tf.float32)
                
                # Gaussian filtre uygulama
                blurred_img = tf.expand_dims(img, axis=0)
                kernel = np.ones(filter_shape + (img.shape[-1],), dtype=np.float32)
                kernel = kernel[:, :, :, tf.newaxis]
                blurred_img = tf.nn.depthwise_conv2d(blurred_img, kernel, strides=[1, 1, 1, 1], padding='SAME')
                blurred_img = tf.squeeze(blurred_img)
                
                # Filtrelenmiş görüntüyü kaydetme
                filtered_image_path = os.path.join(output_class_path, image_file)
                tf.keras.preprocessing.image.save_img(filtered_image_path, blurred_img.numpy())

# Train, test ve val dizinlerindeki görüntülere düşük geçişli filtre uygulanması
for directory in ['train', 'test', 'val']:
    input_path = os.path.join(input_directory, directory)
    output_path = os.path.join(output_directory, directory + '_lowpass')
    
    apply_low_pass_filter(input_path, output_path)
