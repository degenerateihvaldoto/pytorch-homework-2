# homework_experiments.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from homework_datasets import CSVDataset
from homework_model_modification import LinearRegressionModel, train_linear_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_with_params(model, train_loader, val_loader, lr, opt_name, epochs=30):
    """
    Вспомогательная функция для тренировки с выбором оптимизатора.
    """
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {opt_name}")

    criterion = nn.MSELoss()
    history = {"val_loss": []}

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch).squeeze(-1)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                out = model(x_batch).squeeze(-1)
                loss = criterion(out, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)

    return history


def run_hyperparameter_experiments():
    """
    Эксперименты с различными значениями гиперпараметров и сохранение графиков.
    """
    logger.info("Запуск исследования влияния гиперпараметров...")
    dataset = CSVDataset("data/regression_data.csv", target_column="target", task="regression")
    os.makedirs("plots", exist_ok=True)

    # Сравнение Learning Rate
    plt.figure(figsize=(8, 5))
    for lr in [0.1, 0.01, 0.001]:
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        model = LinearRegressionModel(in_features=dataset.X_tensor.shape[1])
        history = train_with_params(model, train_loader, val_loader, lr=lr, opt_name="SGD")
        plt.plot(history["val_loss"], label=f"LR = {lr}")
    plt.title("Влияние Learning Rate на Validation Loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/exp_learning_rate.png")
    plt.close()

    # Сравнение Batch Size
    plt.figure(figsize=(8, 5))
    for bs in [16, 32, 64]:
        train_loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=bs, shuffle=False)
        model = LinearRegressionModel(in_features=dataset.X_tensor.shape[1])
        history = train_with_params(model, train_loader, val_loader, lr=0.01, opt_name="SGD")
        plt.plot(history["val_loss"], label=f"Batch Size = {bs}")
    plt.title("Влияние Batch Size на Validation Loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/exp_batch_size.png")
    plt.close()

    # Сравнение Оптимизаторов
    plt.figure(figsize=(8, 5))
    for opt_name in ["SGD", "Adam", "RMSprop"]:
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        model = LinearRegressionModel(in_features=dataset.X_tensor.shape[1])
        history = train_with_params(model, train_loader, val_loader, lr=0.01, opt_name=opt_name)
        plt.plot(history["val_loss"], label=f"Оптимизатор = {opt_name}")
    plt.title("Влияние Оптимизатора на Validation Loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/exp_optimizers.png")
    plt.close()

    logger.info("Графики экспериментов с гиперпараметрами сохранены в папку 'plots/'")


def apply_feature_engineering(csv_path: str, target_col: str) -> pd.DataFrame:
    """
    Генерация дополнительных признаков: полиномы, попарные произведения и статистики строк.
    """
    df = pd.read_csv(csv_path)
    features = df.drop(columns=[target_col])
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    df_engineered = df.copy()

    # Создание квадратов признаков
    for col in num_cols:
        df_engineered[f"{col}_sq"] = df_engineered[col] ** 2

    # Создание парных произведений
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col1, col2 = num_cols[i], num_cols[j]
            df_engineered[f"{col1}_x_{col2}"] = df_engineered[col1] * df_engineered[col2]

    # Вычисление агрегированных статистик строк
    df_engineered["row_mean"] = df_engineered[num_cols].mean(axis=1)
    df_engineered["row_std"] = df_engineered[num_cols].std(axis=1).fillna(0)

    return df_engineered


def run_feature_engineering_experiment():
    """
    Сравнение процесса обучения на исходных признаках и усовершенствованных признаках.
    """
    logger.info("Запуск эксперимента Feature Engineering...")
    df_eng = apply_feature_engineering("data/regression_data.csv", "target")
    df_eng.to_csv("data/regression_engineered.csv", index=False)

    base_dataset = CSVDataset("data/regression_data.csv", "target", "regression")
    eng_dataset = CSVDataset("data/regression_engineered.csv", "target", "regression")

    base_loader = DataLoader(base_dataset, batch_size=32, shuffle=True)
    eng_loader = DataLoader(eng_dataset, batch_size=32, shuffle=True)

    # Обучение базовой модели
    base_model = LinearRegressionModel(in_features=base_dataset.X_tensor.shape[1])
    base_history = train_linear_model(base_model, base_loader, base_loader, epochs=50, lr=0.01)

    # Обучение усовершенствованной модели
    eng_model = LinearRegressionModel(in_features=eng_dataset.X_tensor.shape[1])
    eng_history = train_linear_model(eng_model, eng_loader, eng_loader, epochs=50, lr=0.01)

    logger.info(f"Итоговый лосс базовой модели: {base_history['train_loss'][-1]:.4f}")
    logger.info(f"Итоговый лосс модели после Feature Engineering: {eng_history['train_loss'][-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(base_history["train_loss"], label="Базовые признаки")
    plt.plot(eng_history["train_loss"], label="Усовершенствованные признаки")
    plt.title("Сравнение сходимости моделей с Feature Engineering")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/exp_feature_engineering.png")
    plt.close()
    logger.info("График сравнения сохранен в 'plots/exp_feature_engineering.png'")


if __name__ == "__main__":
    run_hyperparameter_experiments()
    run_feature_engineering_experiment()