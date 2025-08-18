# FlowerIT
from google.colab import drive
drive.mount('/content/drive')

# ==== 0) 제너레이터 준비 (이미 있다면 이 블록은 건너뛰어도 됨) ====
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG = (224, 224); BATCH = 32
train_datagen = ImageDataGenerator(rescale=1./255)  # ※ /255 전처리 그대로
val_datagen   = ImageDataGenerator(rescale=1./255)

train_dir = '/content/drive/MyDrive/TS/dataset/train'
val_dir   = '/content/drive/MyDrive/TS/dataset/val'

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG, batch_size=BATCH, class_mode='categorical', shuffle=True)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=IMG, batch_size=BATCH, class_mode='categorical', shuffle=False)

# ==== 1) class_weight 자동 계산 ====
import numpy as np
ci = train_generator.class_indices                 # {'healthy':0,'sick':1}
labels = train_generator.classes                   # 각 샘플의 정수 라벨
num_classes = len(ci)
counts = np.bincount(labels, minlength=num_classes)
N = counts.sum()
class_weight = {i: (N / (num_classes * counts[i])) for i in range(num_classes)}
print("클래스 순서:", sorted(ci, key=ci.get))
print("클래스별 개수:", counts.tolist())
print("class_weight:", class_weight)

# ==== 2) 모델: 증강(위치/확대) + 백본(ResNet50, /255 입력) ====
import tensorflow as tf
from tensorflow.keras import layers, models

data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),          # ±~5°
    layers.RandomTranslation(0.12, 0.12), # 위치 이동(중심 의존↓)
    layers.RandomZoom(0.0, 0.25),         # 더 크게 보이게(ROI 강조)
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
], name="aug")

base = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=IMG+(3,))
base.trainable = False  # Stage 1: 특징추출

inp = layers.Input(IMG+(3,))
x = data_aug(inp)       # ※ 제너레이터에서 /255 했으므로 여기선 추가 스케일링 없음
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(2, activation='softmax')(x)  # 2클래스
model = models.Model(inp, out)

# ==== 3) Stage 1: 특징추출(보수적 LR) ====
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=7e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=8,
    class_weight=class_weight,          # ← 자동 계산값 사용
    callbacks=[es, rlr]
)

# ==== 4) Stage 2: 파인튜닝(일부 레이어만 해제, BN은 고정) ====
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
    class_weight=class_weight,          # ← 계속 사용
    callbacks=[es, rlr]
)

# ==== 5) 최종 검증: test 세트 정확도 ====
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

TEST_DIR = '/content/drive/MyDrive/TS/dataset/test'  # test/healthy, test/sick
test_datagen = ImageDataGenerator(rescale=1./255)

# 훈련 때의 라벨 순서를 그대로 강제
class_names = sorted(ci, key=ci.get)  # 예: ['healthy', 'sick']

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG,
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False,               # 평가이므로 False
    classes=class_names          # ★ 훈련과 동일 인덱스 순서 강제
)

# 1) 전체 정확도/손실
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print("\n===== [TEST] 전체 =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc : {test_acc*100:.2f}%")

# 2) 클래스별 정확도
test_generator.reset()
probs = model.predict(test_generator, verbose=0)   # (N, num_classes)
y_pred = np.argmax(probs, axis=1)                  # 예측 라벨(정수)
y_true = test_generator.classes                    # 실제 라벨(정수)

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

# 3) 혼동행렬 / 분류 리포트
cm = confusion_matrix(y_true, y_pred)
print("\n[Confusion Matrix]\n", cm)

print("\n[Classification Report]\n",
      classification_report(y_true, y_pred, target_names=class_names, digits=4))

# (선택) 예측 분포 참고
pred_counts = {class_names[i]: int((y_pred==i).sum()) for i in range(len(class_names))}
print("\n예측 분포:", pred_counts)
