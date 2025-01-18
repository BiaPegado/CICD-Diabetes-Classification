import pandas as pd
from sklearn.utils import resample
import pytest

def balance_classes(df, target_column):
    class_majority = df[df[target_column] == 0]
    class_minority = df[df[target_column] == 1]

    class_majority_downsampled = resample(
        class_majority,
        replace=False,
        n_samples=len(class_minority),
        random_state=42
    )

    balanced_df = pd.concat([class_majority_downsampled, class_minority])
    return balanced_df

# Testes
@pytest.fixture
def sample_data():
    dataset_path = "Data/dataset_balanceado.csv"  
    df = pd.read_csv(dataset_path)

    sample = df.groupby('diabetes', group_keys=False).apply(lambda x: x.sample(min(len(x), 4), random_state=42))

    return sample

def test_balance_classes(sample_data):
    balanced_df = balance_classes(sample_data, 'diabetes')

    class_counts = balanced_df['diabetes'].value_counts()
    assert class_counts[0] == class_counts[1], "As classes não estão balanceadas"
    assert len(balanced_df) == len(sample_data), "O tamanho do dataframe não está correto"

def test_data_structure(sample_data):
    expected_columns = {
        'gender', 'age', 'hypertension', 'heart_disease', 
        'smoking_history', 'bmi', 'HbA1c_level', 
        'blood_glucose_level', 'diabetes'
    }
    assert set(sample_data.columns) == expected_columns, "Colunas do dataset estão incorretas"

def test_no_missing_values(sample_data):
    assert sample_data.isnull().sum().sum() == 0, "Existem valores ausentes no dataset"

def test_glucose_values(sample_data):
    assert sample_data['blood_glucose_level'].between(69, 500).all(), "Valores de glicose fora do intervalo esperado"

def test_bmi_values(sample_data):
    assert sample_data['bmi'].between(9, 150).all(), "Valores de BMI fora do intervalo esperado"

def test_hypertension_values(sample_data):
    assert sample_data['hypertension'].between(0, 1).all(), "Valores de hipertensão fora do intervalo esperado"
