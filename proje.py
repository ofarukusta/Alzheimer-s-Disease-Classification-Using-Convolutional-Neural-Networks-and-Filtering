# Gerekli kütüphanelerin yüklenmesi
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import keras.backend as K

# Veri yolu ve diğer parametreler

# ORİJİNAL DATASET#
"""
train_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s Dataset\\train'
test_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s Dataset\\test'
val_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s Dataset\\val'
"""
# LOWPASS DATASET#
"""
train_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetLowPass\\train_lowpass'
test_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetLowPass\\test_lowpass'
val_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetLowPass\\val_lowpass'
"""
# HIGHPASS DATASET#
"""
train_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetHighPass\\train_highpass'
test_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetHighPass\\test_highpass'
val_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetHighPass\\val_highpass'
"""
# MEDIAN FILTER #
"""
train_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetMedianFilter\\train_median'
test_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetMedianFilter\\test_median'
val_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetMedianFilter\\val_median'
"""
# HISTOGRAM FILTER#
train_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetHistogramEqualization\\train_hist_equalized'
test_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetHistogramEqualization\\test_hist_equalized'
val_path = 'C:\\Users\\ofaru\\OneDrive\\Masaüstü\\DSP_Proje\\Alzheimer_s DatasetHistogramEqualization\\val_hist_equalized'


img_size = (150, 150)
batch_size = 32
# Veri yükleme ve ön işleme
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical')

# CNN modeli oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(4, activation='softmax')  # 4 sınıf için softmax aktivasyonu
])


"""
Eğitimi Gerçekleyeceğimiz Train setinden her bir evreye ait örneklerin gösterilmesi
"""
train_batch = next(iter(train_generator))

class_labels = train_generator.class_indices 
class_count = 1  # Her bir sınıftan göstereceğimiz örnek sayısı
# classes_to_display = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'] #Orijinal Dataset
# classes_to_display = ['MildDemented_lowpass', 'ModerateDemented_lowpass', 'NonDemented_lowpass', 'VeryMildDemented_lowpass'] #LowPass Dataset
# classes_to_display = ['MildDemented_highPass', 'ModerateDemented_highPass', 'NonDemented_highPass', 'VeryMildDemented_highPass'] #HighPass Dataset
#classes_to_display = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'] #Median Dataset
classes_to_display = ['MildDemented_hist_equalized', 'ModerateDemented_hist_equalized', 'NonDemented_hist_equalized','VeryMildDemented_hist_equalized'] #Histogram Dataset

# Görselleri gösterme
for class_name in classes_to_display:
    class_index = class_labels[class_name]
    found_count = 0
    
    for i in range(len(train_generator.filenames)):
        if train_generator.classes[i] == class_index:
            img = plt.imread(train_path + '\\' + train_generator.filenames[i])
            
            plt.imshow(img)
            plt.title(class_name)
            plt.axis('off')
            plt.show()
            
            found_count += 1
            if found_count == class_count:
                break


# F1 SCORE Fonksiyonu
def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
# Model derleme

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc'),
        f1_score,
]
model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=METRICS
    )

# Modelin eğitimi
num_epochs = 15
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator  # Validation setini burada ekledik
)


# Model performansını görselleştirme
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test veri setinden örnekler alarak modelinizi değerlendirme
test_batch = next(iter(test_generator))
#test_loss, test_accuracy = model.evaluate(test_generator)
#print(f"Test Accuracy: {test_accuracy}")


# Test veri setinden bazı örnekler üzerine tahmin yaptırma
for i in range(4):
    img = test_batch[0][i]  # Görüntü
    true_label_index = np.argmax(test_batch[1][i])  # Gerçek etiketin indeksi
    
    # Görüntüyü model üzerinde tahmin etme
    predictions = model.predict(np.expand_dims(img, axis=0))
    predicted_label_index = np.argmax(predictions)  # Tahmin edilen etiketin indeksi
    
    # Sınıf etiketlerinin adlarını alma
    class_labels = {v: k for k, v in test_generator.class_indices.items()}
    
    # Tahmin edilen ve gerçek etiketlerin adlarını alıp yazdırma
    predicted_label = class_labels[predicted_label_index]
    true_label = class_labels[true_label_index]
    
    # Görüntüyü gösterme ve tahmin edilen sınıfı ekrana yazdırma
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}\nTrue Label: {true_label}")
    plt.axis('off')
    plt.show()

def Train_Val_Plot(acc,val_acc,loss,val_loss,auc,val_auc,precision,val_precision,f1,val_f1):
    
    fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize= (20,5))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['training', 'validation'])


    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['training', 'validation'])
    
    ax3.plot(range(1, len(auc) + 1), auc)
    ax3.plot(range(1, len(val_auc) + 1), val_auc)
    ax3.set_title('History of AUC')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('AUC')
    ax3.legend(['training', 'validation'])
    
    ax4.plot(range(1, len(precision) + 1), precision)
    ax4.plot(range(1, len(val_precision) + 1), val_precision)
    ax4.set_title('History of Precision')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Precision')
    ax4.legend(['training', 'validation'])
    
    ax5.plot(range(1, len(f1) + 1), f1)
    ax5.plot(range(1, len(val_f1) + 1), val_f1)
    ax5.set_title('History of F1-score')
    ax5.set_xlabel('Epochs')
    ax5.set_ylabel('F1 score')
    ax5.legend(['training', 'validation'])


    plt.show()
    

Train_Val_Plot(history.history['accuracy'],history.history['val_accuracy'],
               history.history['loss'],history.history['val_loss'],
               history.history['auc'],history.history['val_auc'],
               history.history['precision'],history.history['val_precision'],
               history.history['f1_score'],history.history['val_f1_score']
              )
