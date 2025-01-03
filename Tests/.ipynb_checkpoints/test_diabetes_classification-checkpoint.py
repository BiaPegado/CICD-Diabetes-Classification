import pandas as pd
from sklearn.utils import resample
import pytest

# Função para balancear classes
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
    # Carregar o dataset completo e pegar uma amostra
    dataset_path = "Data/dataset_balanceado.csv"  
    df = pd.read_csv(dataset_path)

    # Garantir que a amostra tenha pelo menos alguns exemplos de cada classe
    sample = df.groupby('diabetes', group_keys=False).apply(lambda x: x.sample(min(len(x), 4), random_state=42))

    return sample

def test_balance_classes(sample_data):
    # Testar se a função balance_classes funciona corretamente
    balanced_df = balance_classes(sample_data, 'diabetes')

    class_counts = balanced_df['diabetes'].value_counts()
    assert class_counts[0] == class_counts[1], "As classes não estão balanceadas"
    assert len(balanced_df) == len(sample_data), "O tamanho do dataframe não está correto"

def test_data_structure(sample_data):
    # Verificar se o dataframe contém as colunas esperadas
    expected_columns = {
        'gender', 'age', 'hypertension', 'heart_disease', 
        'smoking_history', 'bmi', 'HbA1c_level', 
        'blood_glucose_level', 'diabetes'
    }
    assert set(sample_data.columns) == expected_columns, "Colunas do dataset estão incorretas"

def test_no_missing_values(sample_data):
    # Certificar-se de que não há valores ausentes
    assert sample_data.isnull().sum().sum() == 0, "Existem valores ausentes no dataset"

def test_glucose_values(sample_data):
    # Verificar se os valores de glicose estão dentro de um intervalo esperado
    assert sample_data['blood_glucose_level'].between(70, 200).all(), "Valores de glicose fora do intervalo esperado"

def test_bmi_values(sample_data):
    # Verificar se os valores de BMI estão dentro de um intervalo razoável
    assert sample_data['bmi'].between(10, 50).all(), "Valores de BMI fora do intervalo esperado"

def test_hypertension_values(sample_data):
    # Verificar se os valores de hipertensão estão dentro de um intervalo razoável
    assert sample_data['hypertension'].between(0, 1).all(), "Valores de hipertensão fora do intervalo esperado"
