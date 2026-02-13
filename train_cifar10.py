"""
CIFAR-10 圖片分類模型訓練腳本
可以辨識 10 種物品：飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船、卡車
"""

print("=" * 60)
print("CIFAR-10 圖片分類模型訓練")
print("=" * 60)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 減少 TensorFlow 警告訊息

try:
    import tensorflow as tf
    from tensorflow import keras
    import tensorflowjs as tfjs
    import numpy as np
    print("✓ 套件載入成功")
except ImportError as e:
    print("✗ 缺少必要套件，請執行:")
    print("pip install tensorflow tensorflowjs numpy")
    exit(1)

# CIFAR-10 類別名稱（中英文）
class_names_en = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
class_names_zh = ['飛機', '汽車', '鳥', '貓', '鹿', 
                  '狗', '青蛙', '馬', '船', '卡車']

print("\n1. 載入 CIFAR-10 資料集...")
try:
    # 載入 CIFAR-10 資料集
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # 正規化到 0-1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 將標籤壓平 (50000, 1) -> (50000,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    print(f"✓ 訓練集: {x_train.shape[0]} 張圖片 (32x32 彩色)")
    print(f"✓ 測試集: {x_test.shape[0]} 張圖片")
    print(f"✓ 類別數量: {len(class_names_zh)} 種")
    
except Exception as e:
    print(f"✗ 資料載入失敗: {e}")
    exit(1)

print("\n2. 建立 CNN 模型...")
try:
    model = keras.Sequential([
        # 輸入層 (32x32x3 彩色圖片)
        keras.layers.Input(shape=(32, 32, 3)),
        
        # 第一組卷積層
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.2),
        
        # 第二組卷積層
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        # 第三組卷積層
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.4),
        
        # 全連接層
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # 使用 Adam 優化器和學習率衰減
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ 模型架構建立完成")
    print(f"✓ 總參數量: {model.count_params():,}")
    model.summary()
    
except Exception as e:
    print(f"✗ 模型建立失敗: {e}")
    exit(1)

print("\n3. 訓練模型...")
print("(這需要 10-20 分鐘，請耐心等待)")
print("-" * 60)

try:
    # 數據增強（提高準確率）
    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)
    
    # 學習率調整
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,
        patience=3, 
        min_lr=0.00001,
        verbose=1
    )
    
    # 早停（防止過擬合）
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # 開始訓練
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )
    
    print("\n✓ 訓練完成！")
    
except Exception as e:
    print(f"✗ 訓練失敗: {e}")
    exit(1)

print("\n4. 評估模型...")
try:
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✓ 測試準確率: {test_acc*100:.2f}%")
    print(f"✓ 測試損失: {test_loss:.4f}")
    
    # 顯示每個類別的準確率
    predictions = model.predict(x_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    print("\n各類別準確率:")
    print("-" * 40)
    for i in range(10):
        mask = y_test == i
        acc = np.mean(pred_classes[mask] == y_test[mask])
        print(f"{class_names_zh[i]:4s} ({class_names_en[i]:10s}): {acc*100:5.2f}%")
    
except Exception as e:
    print(f"✗ 評估失敗: {e}")

print("\n5. 儲存模型...")
try:
    # 儲存 Keras 模型
    model.save('cifar10_model.h5')
    print("✓ Keras 模型已儲存: cifar10_model.h5")
except Exception as e:
    print(f"⚠ Keras 模型儲存失敗: {e}")

print("\n6. 轉換為 TensorFlow.js 格式...")
try:
    # 轉換為 TensorFlow.js
    tfjs.converters.save_keras_model(model, 'cifar10_tfjs_model')
    print("✓ TensorFlow.js 模型已儲存: cifar10_tfjs_model/")
    
    # 檢查檔案
    import os
    files = os.listdir('cifar10_tfjs_model')
    print("\n生成的檔案:")
    total_size = 0
    for f in files:
        size = os.path.getsize(f'cifar10_tfjs_model/{f}')
        total_size += size
        print(f"  - {f} ({size/1024:.2f} KB)")
    print(f"\n總大小: {total_size/1024/1024:.2f} MB")
    
except Exception as e:
    print(f"✗ 轉換失敗: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✓✓✓ 訓練完成！✓✓✓")
print("=" * 60)
print("\n下一步:")
print("1. 建立新資料夾: cifar10-app")
print("2. 在裡面建立 model/ 子資料夾")
print("3. 將 cifar10_tfjs_model/ 中的所有檔案複製到 model/")
print("4. 下載網頁應用程式檔案（稍後提供）")
print("5. 啟動伺服器測試:")
print("   cd cifar10-app")
print("   python -m http.server 8000")
print("\n預期準確率: 75-85%")
print("=" * 60)
