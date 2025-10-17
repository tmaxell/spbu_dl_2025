import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# --- 0. Настройки и определение устройства ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {DEVICE}")

# --- 1. Реализация кастомного оптимизатора ---
class CustomSGD(torch.optim.Optimizer):
    """
    Реализация стохастического градиентного спуска с моментумом.
    """
    def __init__(self, params, lr=1e-3, momentum=0.9):
        if lr < 0.0:
            raise ValueError(f"Некорректная скорость обучения: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Некорректное значение моментума: {momentum}")

        defaults = dict(lr=lr, momentum=momentum)
        super(CustomSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomSGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # Инициализация буфера моментума, если его нет
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = state['momentum_buffer']
                    # Обновление моментума: v = beta * v + grad
                    buf.mul_(momentum).add_(grad)

                # Обновление весов: w = w - lr * v
                p.add_(state['momentum_buffer'], alpha=-lr)
                
        return loss

# --- 2. Архитектура модели (MLP) ---
class SongYearPredictor(nn.Module):
    def __init__(self, input_features=90, dropout_rate=0.3):
        super(SongYearPredictor, self).__init__()
        self.network = nn.Sequential(
            # Блок 1
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Блок 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Блок 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Выходной слой
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# --- 3. Загрузка и предобработка данных ---
try:
    train_x_df = pd.read_csv('data/train_x.csv', index_col=0)
    train_y_df = pd.read_csv('data/train_y.csv', index_col=0)
    test_x_df = pd.read_csv('data/test_x.csv', index_col='id')
except FileNotFoundError:
    print("Ошибка: Файлы не найдены.")
    exit()

X = train_x_df.values
y = train_y_df.values
X_test = test_x_df.values

# Масштабирование признаков
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)
X_test_scaled = feature_scaler.transform(X_test)

# Масштабирование цели
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y)

# Конвертация в тензоры PyTorch
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)


# --- 4. Цикл обучения и оценки ---
def train_model(optimizer_name='custom_sgd'):
    
    print(f"\n--- Начинаем обучение с оптимизатором: {optimizer_name} ---")
    
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros_like(y_scaled)
    test_predictions = np.zeros((X_test_tensor.shape[0], 1))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor, y_tensor)):
        print(f"--- Фолд {fold + 1}/{N_SPLITS} ---")
        
        # Разделение данных
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
        
        # Создание DataLoader'ов
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=1024)

        # Инициализация модели, критерия и оптимизатора
        model = SongYearPredictor().to(DEVICE)
        criterion = nn.MSELoss()
        
        if optimizer_name == 'custom_sgd':
            optimizer = CustomSGD(model.parameters(), lr=0.005, momentum=0.9)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            raise ValueError("Неизвестное имя оптимизатора")
        
        # Ранняя остановка
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 7

        for epoch in range(100): # Максимум 100 эпох
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Валидация
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)

            if (epoch + 1) % 10 == 0:
                 print(f"Эпоха {epoch+1}, Валидационная MSE: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Сохраняем лучшую модель на фолде
                torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Ранняя остановка на эпохе {epoch+1}")
                    break
        
        # Загрузка лучшей модели и предсказание
        model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
        model.eval()
        with torch.no_grad():
            oof_predictions[val_idx] = model(X_val.to(DEVICE)).cpu().numpy()
            test_predictions += model(X_test_tensor.to(DEVICE)).cpu().numpy() / N_SPLITS

    # Оценка OOF (Out-of-Fold)
    final_oof_predictions = target_scaler.inverse_transform(oof_predictions)
    oof_mse = mean_squared_error(y, final_oof_predictions)
    
    print(f"\nРезультат для оптимизатора '{optimizer_name}':")
    print(f"Итоговый OOF MSE по кросс-валидации: {oof_mse:.4f}")
    
    return test_predictions, oof_mse

# --- 5. Запуск обучения и сравнение ---
custom_sgd_preds, custom_sgd_mse = train_model(optimizer_name='custom_sgd')
adam_preds, adam_mse = train_model(optimizer_name='adam')

print("\n--- Сравнение оптимизаторов ---")
print(f"MSE с CustomSGD: {custom_sgd_mse:.4f}")
print(f"MSE с Adam:      {adam_mse:.4f}")

if custom_sgd_mse < adam_mse:
    print("Кастомный оптимизатор показал лучший результат!")
    final_test_predictions_scaled = custom_sgd_preds
else:
    print("Стандартный Adam показал лучший результат. Используем его для сабмишена.")
    final_test_predictions_scaled = adam_preds
    
# --- 6. Формирование submission файла ---
# Обратное преобразование предсказаний
final_predictions = target_scaler.inverse_transform(final_test_predictions_scaled)
final_predictions_rounded = np.round(final_predictions).astype(int).flatten()

# Создание DataFrame
submission = pd.DataFrame({
    'id': test_x_df.index,
    'year': final_predictions_rounded
})

submission.to_csv('submission.csv', index=False)

print("\nФайл 'submission.csv' успешно создан.")
print(submission.head())