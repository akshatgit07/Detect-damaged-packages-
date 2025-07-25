import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Step 1: Load Valid CSVs ===
csv_dir = '/tmp/preprocessed_data'
csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
             if f.endswith('.csv') and os.path.getsize(os.path.join(csv_dir, f)) > 0]

if not csv_files:
    raise ValueError("No valid CSV files found.")

df = pd.concat([pd.read_csv(f, header=None, names=["path", "label"]) for f in csv_files], ignore_index=True)
df.dropna(inplace=True)

# === Step 2: Load & Preprocess Images ===
def load_image(path):
    try:
        img = Image.open(path).convert('RGB').resize((224, 224))
        return np.array(img)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

images, labels = [], []
for i, row in df.iterrows():
    img = load_image(row['path'])
    if img is not None:
        images.append(img)
        labels.append(1 if row['label'].strip().lower() == 'damaged' else 0)

X = np.array(images)
y = np.array(labels)
print(f"✅ Loaded {len(X)} images")

# === Step 3: Train/Val/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# === Step 4: Class Weight Calculation ===
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train)
class_weights = dict(enumerate(class_weights))
print("📊 Class Weights:", class_weights)

# === Step 5: Data Augmentation Block ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# === Step 6: Define Model ===
model = tf.keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# === Step 7: Train Model with EarlyStopping ===
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# === Step 8: Evaluate ===
loss, acc, prec, rec = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}")

# === Step 9: Save Model ===
model.save("parcel_model_augmented.h5")
print("✅ Saved model to parcel_model_augmented.h5")

# === Step 10: Confusion Matrix ===
y_pred = (model.predict(X_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Intact", "Damaged"])
disp.plot(cmap="Blues")
plt.title("📦 Confusion Matrix (Augmented)")
plt.show()

