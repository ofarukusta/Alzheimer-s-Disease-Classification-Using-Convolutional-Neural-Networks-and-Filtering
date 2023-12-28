import os
import tensorflow as tf

input_directory = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s Dataset'
output_directory = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetHighPass'

def apply_high_pass_filter(input_path, output_path):
    for class_folder in os.listdir(input_path):
        class_path = os.path.join(input_path, class_folder)
        output_class_path = os.path.join(output_path, class_folder + '_highPass')
        
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        for image_file in os.listdir(class_path):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(class_path, image_file)
                img = tf.io.read_file(image_path)
                img = tf.image.decode_image(img, channels=3)
                img = tf.image.convert_image_dtype(img, tf.float32)
                
                # Görüntüdeki kenarları tespit etme
                edges = tf.image.sobel_edges(tf.expand_dims(img, axis=0))
                
                # Dikey ve yatay kenarları kaydetme
                file_name = os.path.basename(image_path)
                
                # Dikey kenarlar
                vertical_edges_path = os.path.join(output_class_path, 'vertical_' + file_name)
                tf.keras.preprocessing.image.save_img(vertical_edges_path, edges[0, ..., 0].numpy())

                # Yatay kenarlar
                horizontal_edges_path = os.path.join(output_class_path, 'horizontal_' + file_name)
                tf.keras.preprocessing.image.save_img(horizontal_edges_path, edges[0, ..., 1].numpy())

# Filtreleme işlemi ve yeni klasörlere kaydetme
for directory in ['train', 'test', 'val']:
    input_path = os.path.join(input_directory, directory)
    output_path = os.path.join(output_directory, directory + '_highpass')
    
    apply_high_pass_filter(input_path, output_path)
