import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
# Використовую код з 1 лабораторної (додала фунції для різних методів відбору за невизначеністю)
# Більшість коментарів які були в лабораторній 1 забрала, для кращої читабельності

import tensorflow as tf

print("Доступні GPU:", tf.config.list_physical_devices('GPU'))

# Функції для обчислення метрик невизначеності
# Метод найменшої впевненості
# Працює як пошук класа з найбільшим значенням впевненості і віднімається від одиниці
# чим більше значення тим більше вони не визначені(самий простий метод)
def least_confidence(probs):
    """Вибір зразків з найменшою впевненістю."""
    return 1 - np.max(probs, axis=1)
# Метод межі
# Логіка в ріхниці впевненості між двома найбільш вірогідними класами
# Чим значення менше тим більше вони не визначені
def margin_sampling(probs):
    """Обчислює різницю між двома найвищими ймовірностями класів."""
    part_sorted = -np.partition(-probs, 2, axis=1)
    return part_sorted[:, 0] - part_sorted[:, 1]
# Метод відношення впевненостей
# Тут вже знаходять відношення між двома найбільш вірогідними класами
# Чим значення менше тим більше вони не визначені
# 1e-10 для того шоб попередити ділення на нуль
def ratio_confidence(probs):
    """Обчислює відношення між ймовірністю найвищого і другого найвищого класів."""
    part_sorted = -np.partition(-probs, 2, axis=1)
    return part_sorted[:, 1] / (part_sorted[:, 0] + 1e-10)
# Метод ентропії
# Використовується формула ентропії
# чим більше значення тим більше вони не визначені
def entropy(probs):
    """Вимірює невизначеність прогнозу за допомогою ентропії."""
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)


def build_dataset(X, y, batch, repeat=True, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
# Додала метод для того щоб опрацьовувати різні стратегії
def select_samples(strategy, probs, sample_size):
    """Функція для вибору зразків за стратегією."""
    if strategy == 'least_confidence':
        scores = least_confidence(probs)
        selected_indices = np.argsort(scores)[-sample_size:]
    elif strategy == 'margin_sampling':
        scores = margin_sampling(probs)
        selected_indices = np.argsort(scores)[:sample_size]
    elif strategy == 'ratio_confidence':
        scores = ratio_confidence(probs)
        selected_indices = np.argsort(scores)[:sample_size]
    elif strategy == 'entropy':
        scores = entropy(probs)
        selected_indices = np.argsort(scores)[-sample_size:]
    else:
        raise ValueError(f"Невідома стратегія: {strategy}")

    return selected_indices

# Задаю параметри активного навчання
num_iterations = 5
sampling_size = 1
acc_baseline = 0.99
STEPS_PER_EPOCH = 100 
BATCH_SIZE = 32

# Підготовка даних
(X, y), (X_test_full, y_test_full) = cifar10.load_data()

# Зменшення датасету до половини
half_index = len(X) // 2  # Індекс для половини даних
X, y = X[:half_index], y[:half_index]

y = y.flatten()
y_test_full = y_test_full.flatten()

# Вибір 3 класів
selected_classes = [0, 1, 2]  
mask = np.isin(y, selected_classes)
X, y = X[mask], y[mask]
y = np.array([np.where(selected_classes == label)[0][0] for label in y])

# Розділення даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, len(selected_classes))
y_test = to_categorical(y_test, len(selected_classes))
# Потім в 6 лабораторній буду змінювати кількість розмічених зразків(зараз 10%)


def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'), 
        Dropout(0.5),
        Dense(len(selected_classes), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model



y_test_classes = np.argmax(y_test, axis=1)



# Змінна для відстеження часу виконання 
execution_time = {}
active_results = {}

strategies = ['least_confidence', 'margin_sampling', 'ratio_confidence', 'entropy']
# Тут змінюю частку данних яку відкладаю на пул
X_initial, X_pool, y_initial, y_pool = train_test_split(X_train, y_train, test_size=0.90, random_state=42, stratify=y_train)

for strategy in strategies:
    print(f"=== Стратегія: {strategy} ===")
    active_model = create_model()
    X_initialStrategy = X_initial
    y_initialStrategy = y_initial
    X_poolStrategy = X_pool
    y_poolStrategy = y_pool
    # Первинне навчання моделі
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    active_model.fit(X_initialStrategy, y_initialStrategy, epochs=5, validation_split=0.1, batch_size=32, callbacks=[checkpoint])
    
    # Оцінка точності до активного навчання
    loss, acc = active_model.evaluate(X_test, y_test, verbose=0)
    
    al_history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'execution_time': [],
        'accuracy_before': [acc],  
        'accuracy_after': []
    }
    
    start_time = time.time()  # Початок вимірювання часу виконання
    val_loss, val_acc = active_model.evaluate(X_test, y_test, verbose=0)
    al_history['val_accuracy'].append(val_acc)
    al_history['loss'].append(loss)
    al_history['val_loss'].append(val_loss)
   
    y_pred = active_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
   
    al_history['precision'].append(precision)
    al_history['recall'].append(recall)
    al_history['f1'].append(f1)
    for iteration in range(num_iterations):
        # Оцінка до відбору
     

        if acc >= acc_baseline:
            break
        # Вибір 5% зразків для активного навчання
        sampling_size = max(1, int(0.05 * len(X_poolStrategy)))
        # Вибір зразків для активного навчання
        y_pool_proba = active_model.predict(X_poolStrategy)
        selected_indices = select_samples(strategy, y_pool_proba, sampling_size)
        
        x_sample = X_poolStrategy[selected_indices]
        y_sample = y_poolStrategy[selected_indices]
        
        y_initialStrategy = np.concatenate((y_initialStrategy, y_sample), axis=0)
        X_initialStrategy = np.concatenate((X_initialStrategy, x_sample), axis=0)
        X_poolStrategy = np.delete(X_poolStrategy, selected_indices, axis=0)
        y_poolStrategy = np.delete(y_poolStrategy, selected_indices, axis=0)
        
        train_dataset = build_dataset(X_initialStrategy, y_initialStrategy, batch=BATCH_SIZE, shuffle=True)
        early_stopping = EarlyStopping(
        monitor='val_accuracy',  # Моніторимо точність на валідації
        patience=2,              # Зупиняємо після 2 епох без покращення
        restore_best_weights=True  # Відновлюємо найкращі ваги
        )
        active_model.fit(
            train_dataset,
            steps_per_epoch=len(X_initialStrategy) // BATCH_SIZE,
            epochs=5,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint, early_stopping]
        )
        active_model.save('best_active_model.keras')
        active_model = tf.keras.models.load_model('best_active_model.keras')
        val_loss, val_acc = active_model.evaluate(X_test, y_test, verbose=0)
        al_history['val_accuracy'].append(val_acc)
        al_history['loss'].append(loss)
        al_history['val_loss'].append(val_loss)
   
        y_pred = active_model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
   
        al_history['precision'].append(precision)
        al_history['recall'].append(recall)
        al_history['f1'].append(f1)
        # Оцінка після відбору
        loss, acc = active_model.evaluate(X_test, y_test, verbose=0)
        print(f"Остання точність після навчання: {acc}")
        al_history['accuracy_after'].append(acc)

    end_time = time.time()  # Кінець вимірювання часу виконання
    total_time = end_time - start_time
    al_history['execution_time'] = total_time

    # Зберігаємо результати для поточної стратегії
    execution_time[strategy] = total_time
    
    active_results[strategy] = al_history

    print(f"Час виконання для стратегії {strategy}: {total_time:.2f} секунд")  
    print(f"Точність до навчання (після первинного навчання): {al_history['accuracy_before'][0]}")
    print(f"Остання точність після навчання: {al_history['accuracy_after'][-1]}")
    if os.path.exists('best_model.keras'):
       os.remove('best_model.keras')
       print("Файл 'best_model.keras' видалено.")


# Вивід результатів
for strategy in strategies:
    print(f"=== Результати для стратегії {strategy} ===")
    print(f"Час виконання (секунди): {execution_time[strategy]:.2f}")
    print(f"Остання точність до навчання: {active_results[strategy]['accuracy_before'][-1]}")
    print(f"Остання точність після навчання: {active_results[strategy]['accuracy_after'][-1]}")

import matplotlib.pyplot as plt

# Функція для побудови графіків для кожної стратегії
def plot_learning_curves(active_results, metric_name, ylabel, title):
    plt.figure(figsize=(10, 6))
    for strategy, history in active_results.items():
        plt.plot(history[metric_name], label=strategy)
    
    plt.xlabel('Ітерації')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Побудова графіків точності
plot_learning_curves(active_results, 'accuracy_after', 'Точність після навчання', 'Точність навчання для кожної стратегії')

# Побудова графіків втрат
plot_learning_curves(active_results, 'val_loss', 'Втрати', 'Втрати під час навчання для кожної стратегії')

# Побудова графіків precision
plot_learning_curves(active_results, 'precision', 'Precision', 'Precision під час навчання для кожної стратегії')

# Побудова графіків recall
plot_learning_curves(active_results, 'recall', 'Recall', 'Recall під час навчання для кожної стратегії')

# Побудова графіків F1-score
plot_learning_curves(active_results, 'f1', 'F1-Score', 'F1-Score під час навчання для кожної стратегії')