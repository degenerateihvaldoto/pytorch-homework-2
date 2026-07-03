import torch
import torch.nn as nn
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LinearRegressionModel(nn.Module):
    """
    Класс линейной регрессии на PyTorch.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MultiClassLogisticRegressionModel(nn.Module):
    """
    Класс логистической регрессии для многоклассовой классификации.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_model(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 0.01,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        patience: int = 10
) -> dict:
    """
    Обучение линейной модели с L1/L2 регуляризацией и Early Stopping.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch).squeeze(-1)
            loss = criterion(outputs, y_batch)

            # Расчет штрафов регуляризации
            if l1_lambda > 0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_penalty
            if l2_lambda > 0:
                l2_penalty = sum((p ** 2).sum() for p in model.parameters())
                loss += l2_lambda * l2_penalty

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch).squeeze(-1)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        # Проверка Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping сработал на эпохе {epoch + 1}")
                model.load_state_dict(best_weights)
                break

    return history


def train_classification_model(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 0.01,
        patience: int = 10
) -> dict:
    """
    Обучение классификатора с использованием CrossEntropyLoss и Early Stopping.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}
    best_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        # Проверка Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping сработал на эпохе {epoch + 1}")
                model.load_state_dict(best_weights)
                break

    return history


def evaluate_classifier(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        num_classes: int
) -> tuple:
    """
    Расчет метрик оценки классификации: Precision, Recall, F1, ROC-AUC и Confusion Matrix.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Расчет метрик
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0
    )

    # Расчет ROC-AUC
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
    except Exception as e:
        logger.warning(f"Не удалось рассчитать ROC-AUC: {e}")
        roc_auc = float('nan')

    cm = confusion_matrix(all_targets, all_preds)
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

    return metrics, cm


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str = "plots/confusion_matrix.png"):
    """
    Визуализация Confusion Matrix и сохранение графика в файл.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Действительные классы')
    plt.xlabel('Предсказанные классы')
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Матрица ошибок сохранена в {save_path}")


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader

    logger.info("=== Запуск демонстрационного обучения ===")

    # 1. Демонстрация Линейной Регрессии с L1/L2 и Early Stopping
    logger.info("--- Тестирование Линейной Регрессии ---")
    np.random.seed(42)
    torch.manual_seed(42)

    # Генерируем тестовые тензоры
    X_reg = torch.randn(150, 5)
    true_w = torch.tensor([[1.5, -2.0, 0.0, 0.5, -1.0]]).T
    y_reg = (X_reg @ true_w).squeeze(-1) + torch.randn(150) * 0.1

    dataset_reg = TensorDataset(X_reg, y_reg)
    train_loader_reg = DataLoader(dataset_reg, batch_size=16, shuffle=True)
    val_loader_reg = DataLoader(dataset_reg, batch_size=16, shuffle=False)

    lin_model = LinearRegressionModel(in_features=5)

    # Запускаем обучение с регуляризацией
    history_reg = train_linear_model(
        lin_model,
        train_loader_reg,
        val_loader_reg,
        epochs=100,
        lr=0.01,
        l1_lambda=0.01,
        l2_lambda=0.01,
        patience=7
    )
    logger.info(f"Линейная модель обучена. Финальный Loss на валидации: {history_reg['val_loss'][-1]:.4f}")

    # 2. Демонстрация Многоклассовой Логистической Регрессии
    logger.info("\n--- Тестирование Логистической Регрессии ---")
    X_clf = torch.randn(200, 4)
    # Создаем 3 класса на основе признаков
    sum_x = X_clf.sum(dim=1)
    y_clf = torch.zeros(200, dtype=torch.long)
    y_clf[sum_x < -0.5] = 0
    y_clf[(sum_x >= -0.5) & (sum_x < 0.5)] = 1
    y_clf[sum_x >= 0.5] = 2

    dataset_clf = TensorDataset(X_clf, y_clf)
    train_loader_clf = DataLoader(dataset_clf, batch_size=16, shuffle=True)
    val_loader_clf = DataLoader(dataset_clf, batch_size=16, shuffle=False)

    clf_model = MultiClassLogisticRegressionModel(in_features=4, num_classes=3)

    # Запускаем классификацию
    train_classification_model(
        clf_model,
        train_loader_clf,
        val_loader_clf,
        epochs=100,
        lr=0.02,
        patience=10
    )

    metrics, cm = evaluate_classifier(clf_model, val_loader_clf, num_classes=3)
    logger.info(f"Метрики качества классификации: {metrics}")

    # Отрисовываем confusion matrix
    plot_confusion_matrix(cm, class_names=["Класс 0", "Класс 1", "Класс 2"],
                          save_path="plots/demo_confusion_matrix.png")
    logger.info("=== Демонстрационный запуск успешно завершен ===")