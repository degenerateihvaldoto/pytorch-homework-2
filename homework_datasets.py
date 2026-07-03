# homework_datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging
from homework_model_modification import (
    LinearRegressionModel,
    MultiClassLogisticRegressionModel,
    train_linear_model,
    train_classification_model,
    evaluate_classifier,
    plot_confusion_matrix
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CSVDataset(Dataset):
    """
    Кастомный класс PyTorch Dataset для автоматизированного импорта данных из CSV-файлов,
    заполнения пропусков, кодирования категорий и масштабирования числовых признаков.
    """

    def __init__(self, csv_file: str, target_column: str, task: str = 'regression', transform=None):
        super().__init__()
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Файл {csv_file} не найден.")

        self.df = pd.read_csv(csv_file)
        self.target_column = target_column
        self.task = task
        self.transform = transform

        if target_column not in self.df.columns:
            raise ValueError(f"Целевая колонка '{target_column}' не найдена в {csv_file}")

        self.y_raw = self.df[target_column]
        self.X_raw = self.df.drop(columns=[target_column])

        # Предобработка признаков
        self.X_processed = self._preprocess_features(self.X_raw)
        self.y_processed = self._preprocess_target(self.y_raw)

        # Преобразование в тензоры PyTorch
        self.X_tensor = torch.tensor(self.X_processed.values, dtype=torch.float32)
        if self.task == 'regression':
            self.y_tensor = torch.tensor(self.y_processed.values, dtype=torch.float32)
        else:
            self.y_tensor = torch.tensor(self.y_processed.values, dtype=torch.long)

    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Заполнение пропусков и нормализация числовых признаков
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())

        if len(num_cols) > 0:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])

        # Заполнение пропусков и One-Hot Encoding категориальных признаков
        for col in cat_cols:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].fillna(mode_val)

        if len(cat_cols) > 0:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Гарантируем преобразование булевых колонок в тип float после get_dummies
        return df.astype(float)

    def _preprocess_target(self, y: pd.Series) -> pd.Series:
        if self.task == 'regression':
            y = y.fillna(y.mean())
            return y
        else:
            # Классификация
            y = y.fillna(y.mode()[0] if not y.mode().empty else 0)
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.classes_ = le.classes_
            self.num_classes = len(le.classes_)
            return pd.Series(y_encoded)

    def __len__(self) -> int:
        return len(self.X_tensor)

    def __getitem__(self, idx: int) -> tuple:
        x = self.X_tensor[idx]
        y = self.y_tensor[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def generate_dummy_data():
    """
    Автоматическая генерация демонстрационных CSV датасетов.
    """
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)
    n_samples = 300

    # Регрессия содержит числовые признаки, одну категорию и таргет
    reg_data = {
        'feature_num_1': np.random.randn(n_samples),
        'feature_num_2': np.random.randn(n_samples) * 5 + 2,
        'feature_cat': np.random.choice(['Group A', 'Group B', 'Group C'], size=n_samples),
        'target': np.random.randn(n_samples) * 3.0
    }
    reg_df = pd.DataFrame(reg_data)
    # Имитируем пропущенные значения для проверки предобработки
    reg_df.loc[::15, 'feature_num_1'] = np.nan
    reg_df.to_csv("data/regression_data.csv", index=False)

    # Классификация
    clf_data = {
        'feature_num_1': np.random.randn(n_samples),
        'feature_num_2': np.random.randn(n_samples) * 2 - 1,
        'feature_cat': np.random.choice(['Low', 'Medium', 'High'], size=n_samples),
        'target': np.random.choice(['Class_0', 'Class_1', 'Class_2'], size=n_samples)
    }
    clf_df = pd.DataFrame(clf_data)
    clf_df.to_csv("data/classification_data.csv", index=False)
    logger.info("Синтетические датасеты сгенерированы в директории 'data/'")


if __name__ == "__main__":
    generate_dummy_data()

    logger.info("=== Старт тестов обучения по батчам на CSV данных ===")

    reg_dataset = CSVDataset("data/regression_data.csv", target_column="target", task="regression")
    reg_loader = DataLoader(reg_dataset, batch_size=32, shuffle=True)

    reg_model = LinearRegressionModel(in_features=reg_dataset.X_tensor.shape[1])
    logger.info("Обучение линейной регрессии с регуляризацией...")
    history_reg = train_linear_model(
        reg_model, reg_loader, reg_loader, epochs=30, lr=0.01, l1_lambda=0.01, l2_lambda=0.01
    )
    logger.info(f"Финальный лосс регрессии: {history_reg['val_loss'][-1]:.4f}")

    clf_dataset = CSVDataset("data/classification_data.csv", target_column="target", task="classification")
    clf_loader = DataLoader(clf_dataset, batch_size=32, shuffle=True)

    clf_model = MultiClassLogisticRegressionModel(
        in_features=clf_dataset.X_tensor.shape[1], num_classes=clf_dataset.num_classes
    )
    logger.info("Обучение классификатора на данных из CSV...")
    train_classification_model(clf_model, clf_loader, clf_loader, epochs=30, lr=0.01)

    metrics, cm = evaluate_classifier(clf_model, clf_loader, num_classes=clf_dataset.num_classes)
    logger.info(f"Метрики классификации: {metrics}")

    plot_confusion_matrix(cm, class_names=list(map(str, clf_dataset.classes_)),
                          save_path="plots/csv_confusion_matrix.png")