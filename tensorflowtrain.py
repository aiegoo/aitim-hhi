import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. Load pre-trained EfficientNet (no top classification layers)
base_model = tf.keras.applications.EfficientNetB1(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)  # Adjust input size if needed
)

# 2. Add your classification head
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.7)(x)  # Add dropout for regularization
predictions = tf.keras.layers.Dense(9, activation='softmax')(x)  # 8 classes
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# 3. Freeze base model layers (optional)
base_model.trainable = False

# 4. Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 5. Data augmentation and loading using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use a portion of the dataset for validation
)

train_generator = train_datagen.flow_from_directory(
    'dataset',  # Path to your dataset
    target_size=(224, 224),  # Match input size
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset',  # Path to your dataset
    target_size=(224, 224),  # Match input size
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 6. Define callbacks for saving the model and early stopping
checkpoint = ModelCheckpoint('best_efficientnet_model.keras', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max', 
                             verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=5, 
                               restore_best_weights=True, 
                               verbose=1)

# 7. Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # Adjust as needed
    callbacks=[checkpoint, early_stopping],
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# 8. Save the trained model
model.save('final_efficientnet_model.h5')

# 9. Optionally, unfreeze some base model layers for fine-tuning
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 40  # Fine-tune last 20 layers

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 10. Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 11. Fine-tune the model
history_fine_tune = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,  # Additional epochs for fine-tuning
    callbacks=[checkpoint, early_stopping],
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# 12. Save the fine-tuned model
model.save('fine_tuned_efficientnet_model.h5')

