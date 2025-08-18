# FlowerIT_ResNet50
from google.colab import drive
drive.mount('/content/drive')


from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG = (224, 224); BATCH = 32
train_datagen = ImageDataGenerator(rescale=1./255) 
val_datagen   = ImageDataGenerator(rescale=1./255)

train_dir = '/content/drive/MyDrive/TS/dataset/train'
val_dir   = '/content/drive/MyDrive/TS/dataset/val'

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG, batch_size=BATCH, class_mode='categorical', shuffle=True)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=IMG, batch_size=BATCH, class_mode='categorical', shuffle=False)

import numpy as np
ci = train_generator.class_indices               
labels = train_generator.classes                   
num_classes = len(ci)
counts = np.bincount(labels, minlength=num_classes)
N = counts.sum()
class_weight = {i: (N / (num_classes * counts[i])) for i in range(num_classes)}
print("클래스 순서:", sorted(ci, key=ci.get))
print("클래스별 개수:", counts.tolist())
print("class_weight:", class_weight)

import tensorflow as tf
from tensorflow.keras import layers, models

data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),        
    layers.RandomTranslation(0.12, 0.12), 
    layers.RandomZoom(0.0, 0.25),       
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
], name="aug")

base = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=IMG+(3,))
base.trainable = False  # Stage 1: 특징추출

inp = layers.Input(IMG+(3,))
x = data_aug(inp)       
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(2, activation='softmax')(x) 
model = models.Model(inp, out)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=7e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=8,
    class_weight=class_weight,      
    callbacks=[es, rlr]
)

for l in base.layers[:-40]:
    l.trainable = False
for l in base.layers[-40:]:
    if isinstance(l, layers.BatchNormalization):
        l.trainable = False
    else:
        l.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=class_weight,      
    callbacks=[es, rlr]
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

TEST_DIR = '/content/drive/MyDrive/TS/dataset/test'  # test/healthy, test/sick
test_datagen = ImageDataGenerator(rescale=1./255)

class_names = sorted(ci, key=ci.get)  # 예: ['healthy', 'sick']

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG,
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False,             
    classes=class_names       
)

test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print("\n===== [TEST] 전체 =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc : {test_acc*100:.2f}%")

test_generator.reset()
probs = model.predict(test_generator, verbose=0)   
y_pred = np.argmax(probs, axis=1)           
y_true = test_generator.classes                  

idx_to_cls = {v:k for k,v in ci.items()}
print("\n라벨 매핑 확인:", ci)

for i, cls in enumerate(class_names):
    m = (y_true == i)
    cnt = int(m.sum())
    corr = int((y_pred[m] == i).sum())
    acc = (corr/cnt*100.0) if cnt>0 else 0.0
    print(f"\n[TEST] {cls}")
    print(f"- 이미지 개수: {cnt}")
    print(f"- 정확도     : {acc:.2f}%")

cm = confusion_matrix(y_true, y_pred)
print("\n[Confusion Matrix]\n", cm)

print("\n[Classification Report]\n",
      classification_report(y_true, y_pred, target_names=class_names, digits=4))

pred_counts = {class_names[i]: int((y_pred==i).sum()) for i in range(len(class_names))}
print("\n예측 분포:", pred_counts)
