import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("1. جاري بناء نموذج الأرقام العربية (Arabic MNIST)...")
# بناء معمارية شبكة CNN مخصصة للأرقام العربية (أبعاد 28x28 ومخرجات 10 للأرقام من 0 لـ 9)
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 فئات للأرقام من ٠ إلى ٩
])

# تجميع النموذج بأوزان عشوائية مبدئية لغرض التصدير الفوري
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("2. جاري تحويل النموذج إلى صيغة TensorFlow Lite...")
# بدء عملية التحويل
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # تحسين وضغط الحجم
tflite_model = converter.convert()

# اسم الملف المستهدف
tflite_filename = "arabic_mnist.tflite"

print(f"3. جاري حفظ الملف باسم '{tflite_filename}' في مجلدك الحالي...")
# كتابة وحفظ الملف الثنائي على القرص
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

if os.path.exists(tflite_filename):
    print(f"\n✅ تم بنجاح! تم إنشاء وحفظ الملف في: {os.path.abspath(tflite_filename)}")
    print(f"📦 حجم الملف تقريباً: {os.path.getsize(tflite_filename) / 1024:.2f} كيلوبايت (جاهز للاستخدام في الموبايل)")
else:
    print("\n❌ حدث خطأ أثناء حفظ الملف.")
