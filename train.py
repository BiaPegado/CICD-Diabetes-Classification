import pandas as pd
import wandb
import params
from pycaret import *

def _create_table(file_path: str, class_labels):
    """
    Cria uma tabela com os parâmetros especificados e adiciona os dados do arquivo CSV.
    """
    # Colunas especificadas
    columns = [str(class_labels[_lab]) for _lab in list(class_labels)]
    table = wandb.Table(columns=["id"] + columns)

    # Ler os dados do arquivo CSV
    data = pd.read_csv(file_path)

    # Adicionar os dados na tabela
    for idx, row in data.iterrows():
      row_id = f"sample_{idx}"
      table.add_data(row_id, *[row[col] for col in columns])

    return table

def save_table_wb(file_path, nome):
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="upload", config=params.CONFIG)
    
    raw_data_at = wandb.Artifact(name=params.RAW_DATA_AT, type="raw_data")
    
    table = _create_table(file_path, params.BDD_CLASSES)
    
    raw_data_at.add(table, nome)
    
    run.log_artifact(raw_data_at)
    run.finish()

save_table_wb('Data/diabetes_prediction_dataset.csv', 'diabetes_prediction__table')

from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


def load_df():
    df = pd.read_csv('Data/diabetes_prediction_dataset.csv')
    return df

def balance_df():
    df = load_df()
    class_majority = df[df['diabetes'] == 0]
    class_minority = df[df['diabetes'] == 1]
    
    class_majority_downsampled = resample(class_majority,
                                          replace=False,
                                          n_samples=len(class_minority),
                                          random_state=42)
    
    df = pd.concat([class_majority_downsampled, class_minority])
    df.to_csv('Data/dataset_balanceado.csv', index=False)
    save_table_wb('Data/dataset_balanceado.csv', "diabetes_prediction__table_balanceada")
    return df

import great_expectations as ge

def set_expectations(df):
    diabetes_expectation_suite = ge.core.ExpectationSuite(
        expectation_suite_name="diabetes_expectation_suite"
    )
    
    diabetes_expectation_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_min_to_be_between",
            kwargs={
                "column": "blood_glucose_level",
                "min_value": 69,
                "max_value": 500,
                "strict_min": True
            }
        )
    )
    
    diabetes_expectation_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "bmi",
                "min_value": 9,
                "max_value": 150
            }
        )
    )
    
    diabetes_expectation_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "hypertension",
                "value_set": [0, 1]
            }
        )
    )
    
    data_asset = ge.from_pandas(df)
    validation_results = data_asset.validate(expectation_suite=diabetes_expectation_suite)
    
    assert validation_results.success, "Interrompendo execução: validação falhou."

from pycaret.classification import *
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import wandb

def upload_metrics_to_wandb(accuracy, recall, precision, f1):
    metrics_table = wandb.Table(columns=["Metric", "Value"])
    metrics_table.add_data("Accuracy", accuracy)
    metrics_table.add_data("Recall", recall)
    metrics_table.add_data("Precision", precision)
    metrics_table.add_data("F1-Score", f1)
    wandb.log({"Model Metrics": metrics_table})

def find_model(df):
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="upload", config=params.CONFIG)
    
    exp = ClassificationExperiment()
    exp.setup(df, target='diabetes', session_id=123)
    
    best = exp.compare_models()
    best_tuned = exp.tune_model(best)
    print(best_tuned)

    exp.plot_model(best, plot='feature', save=True)
    wandb.log({"feature_importance": wandb.Image('Feature Importance.png')})
    
    final_model = exp.finalize_model(best_tuned)
    exp.save_model(final_model, 'Model/diabetes_model')
    
    evaluation_results = exp.evaluate_model(final_model)
    
    filename = "confusion_matrix.png"
    exp.plot_model(final_model, plot="confusion_matrix", save=True)
    wandb.log({"Matriz de Confusão": wandb.Image('Confusion Matrix.png')})
    
    predictions = exp.predict_model(final_model)
    
    y_true = predictions['diabetes'] 
    y_pred = predictions['prediction_label'] 
    
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    upload_metrics_to_wandb(accuracy, recall, precision, f1)

    with open('Results/metrics.txt', 'w') as f:
        f.write(f"Acuracia do modelo: {round(accuracy, 3)}\n")
        f.write(f"Recall: {round(recall, 3)}\n")
        f.write(f"Precisão: {round(precision, 3)}\n")
        f.write(f"F1-Score: {round(f1, 3)}\n")
    
    model_filename = 'Model/diabetes_model.pkl'
    artifact = wandb.Artifact('diabetes_model', type='model')
    artifact.add_file(model_filename)
    wandb.log_artifact(artifact)
    
    wandb.finish()

def build_model():
    df = balance_df()
    set_expectations(df)
    find_model(df)

build_model()
