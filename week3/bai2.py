import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras import regularizers

# ====================================================
# 1. LOAD CIFAR-10 VÀ LẤY 5 LỚP
# ====================================================
(x_train, y_train), _ = cifar10.load_data()
selected_classes = [0, 1, 2, 3, 4]  # airplane–automobile–bird–cat–deer

X_raw, Y_raw = [], []
for cls in selected_classes:
    idx = np.where(y_train.flatten() == cls)[0][:1000]
    X_raw.append(x_train[idx])
    Y_raw.append(y_train[idx])

X_raw = np.concatenate(X_raw)
Y_raw = np.concatenate(Y_raw)

perm = np.random.permutation(len(X_raw))
X_raw, Y_raw = X_raw[perm], Y_raw[perm]

print(f"[+] Selected data: {X_raw.shape}, labels: {Y_raw.shape}")

# ====================================================
# 2. DATA AUGMENTATION CHUẨN
# ====================================================
image_size = X_raw.shape[1]
data_augmentation = tf.keras.Sequential([
    # Zero Padding và Random Crop
    layers.ZeroPadding2D(padding=((4, 4), (4, 4))), 
    layers.RandomCrop(image_size, image_size),      

    # Các phép biến đổi khác
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.001), 
    layers.RandomBrightness(0.001),
])

# ====================================================
# 3. LƯU ẢNH MINH HỌA (ĐÃ FIX DIMENSION)
# ====================================================
def augment_single_image(img):
    img = tf.cast(img, tf.float32) / 255.0
    
    img = tf.expand_dims(img, axis=0) 
    
    augmented_img = data_augmentation(img)
    
    return tf.squeeze(augmented_img, axis=0)

os.makedirs("aug_samples", exist_ok=True)
for i in range(5):
    original = X_raw[i]
    augmented = augment_single_image(original).numpy() 
    
    plt.imsave(f"aug_samples/original_{i}.png", original)
    augmented = np.clip(augmented, 0, 1)
    plt.imsave(f"aug_samples/augmented_{i}.png", augmented)

print("[+] Saved augmented samples -> folder aug_samples/")
# ====================================================
# 4. DATA PIPELINE
# ====================================================
def make_dataset(X, y, batch=64, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(len(X))

    def map_func(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        if augment:
            img = tf.expand_dims(img, axis=0) 
            
            img = data_augmentation(img)
            
            img = tf.squeeze(img, axis=0)
            
        return img, label

    ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# ====================================================
# 5. MÔ HÌNH CNN (2 BLOCKS ĐƠN GIẢN HÓA & L2/BN)
# ====================================================
def build_model():
    # Mức L2 penalty được tăng nhẹ để chống overfitting mạnh hơn
    l2_reg = regularizers.l2(0.005) 
    model = models.Sequential()
    model.add(layers.Input((32, 32, 3)))

    # ================= BLOCK 1 (32 filters) =================
    model.add(layers.Conv2D(32, (3,3), padding='same', use_bias=False, kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2D(32, (3,3), padding='same', use_bias=False, kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.2)) # Giữ mức nhẹ

    # ================= BLOCK 2 (64 filters) =================
    model.add(layers.Conv2D(64, (3,3), padding='same', use_bias=False, kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2D(64, (3,3), padding='same', use_bias=False, kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.35)) # Tăng nhẹ Dropout

    # KHỐI 3 (128 FILTERS) ĐÃ BỊ LOẠI BỎ

    # ================= CLASSIFICATION HEAD =================
    model.add(layers.GlobalAveragePooling2D())
    
    model.add(layers.Dense(256, use_bias=False, kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(len(selected_classes), activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# ====================================================
# 6. TRAIN 1 LẦN (ĐÃ THÊM REDUCELRONPLATEAU)
# ====================================================
def train_once(use_aug):
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, 
        Y_raw, 
        test_size=0.2, 
        stratify=Y_raw 
    )

    train_ds = make_dataset(X_train, y_train, augment=use_aug)
    val_ds = make_dataset(X_val, y_val, augment=False)

    model = build_model()
    # Tăng patience lên 10 để cho phép ReduceLR có thời gian hoạt động
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    )
    # Giảm LR khi val_loss chững lại
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=100, 
        callbacks=[early, reduce_lr],
        verbose=1
    )
    
    return history

# ====================================================
# 7. TRAIN 3 LẦN & LƯU HISTORY
# ====================================================
runs = 3
histories_no_aug = []
histories_aug = []

print(f"\n===== TRAINING {runs} RUNS (NO AUGMENTATION) =====")
for i in range(runs):
    print(f"Running No-Aug {i+1}/{runs}...")
    h = train_once(use_aug=False)
    histories_no_aug.append(h)

print(f"\n===== TRAINING {runs} RUNS (WITH AUGMENTATION) =====")
for i in range(runs):
    print(f"Running With-Aug {i+1}/{runs}...")
    h = train_once(use_aug=True)
    histories_aug.append(h)

# ====================================================
# 8. TÍNH TOÁN KẾT QUẢ VÀ VẼ BIỂU ĐỒ (ĐÃ THÊM LƯU ẢNH)
# ====================================================

# --- 8.1 Tính Average Accuracy ---
acc_no_aug = [h.history['val_accuracy'][-1] for h in histories_no_aug]
acc_aug = [h.history['val_accuracy'][-1] for h in histories_aug]

print("\n===== FINAL RESULTS =====")
print(f"No Augmentation Accuracy:   {np.mean(acc_no_aug):.4f} ± {np.std(acc_no_aug):.4f}")
print(f"With Augmentation Accuracy: {np.mean(acc_aug):.4f} ± {np.std(acc_aug):.4f}")

# --- 8.2 Vẽ biểu đồ Loss (ĐÃ THÊM LƯU ẢNH) ---
def plot_loss_comparison(hist_no, hist_yes):
    os.makedirs("results", exist_ok=True)
    
    plt.figure(figsize=(14, 6))

    # Subplot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.title("Training Loss (3 Runs)")
    for i, h in enumerate(hist_no):
        plt.plot(h.history['loss'], color='red', alpha=0.3, linestyle='-', label='No Aug' if i==0 else "")
    for i, h in enumerate(hist_yes):
        plt.plot(h.history['loss'], color='blue', alpha=0.3, linestyle='-', label='With Aug' if i==0 else "")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Validation Loss
    plt.subplot(1, 2, 2)
    plt.title("Validation Loss (3 Runs)")
    for i, h in enumerate(hist_no):
        plt.plot(h.history['val_loss'], color='red', alpha=0.5, linestyle='--', label='No Aug' if i==0 else "")
    for i, h in enumerate(hist_yes):
        plt.plot(h.history['val_loss'], color='blue', alpha=0.5, linestyle='--', label='With Aug' if i==0 else "")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # LƯU ẢNH VÀO THƯ MỤC results
    plt.savefig("results/loss_comparison.png")
    plt.show()

print("\n[+] Plotting Loss Charts...")
plot_loss_comparison(histories_no_aug, histories_aug)
