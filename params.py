WANDB_PROJECT = "diabetes-classification"
ENTITY = None 
BDD_CLASSES = {i:c for i,c in enumerate(['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
       'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes'])}
RAW_DATA_AT = 'diabetes'
PROCESSED_DATA_AT = 'diabetes_processed'
CONFIG = {
        "model": "diabetes_model",
        "dataset": "Diabetes dataset",
        "metric": "Acurácia"
    }