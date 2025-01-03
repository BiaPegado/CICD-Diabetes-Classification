import pandas as pd
from pycaret import *

df = pd.read_csv('Data/diabetes_prediction_dataset.csv')

df.info()

class_counts = df['diabetes'].value_counts()

print("\nContagem das Classes na Coluna 'diabetes':")
print(f"Classe 0 (Não diabético): {class_counts[0]}")
print(f"Classe 1 (Diabético): {class_counts[1]}")

from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

class_majority = df[df['diabetes'] == 0]
class_minority = df[df['diabetes'] == 1]

class_majority_downsampled = resample(class_majority,
                                      replace=False,
                                      n_samples=len(class_minority),
                                      random_state=42)

df = pd.concat([class_majority_downsampled, class_minority])

from pycaret.classification import *

exp = ClassificationExperiment()

exp.setup(df, target='diabetes', session_id=123)

best = exp.compare_models()
best_tuned = exp.tune_model(best)
print(best_tuned)

exp.plot_model(best, plot = 'feature')

import shutil
import os
from sklearn.metrics import accuracy_score

final_model = exp.finalize_model(best_tuned)
exp.save_model(final_model, 'Model/diabetes_model')

evaluation_results = exp.evaluate_model(final_model)

predictions = exp.predict_model(final_model)
y_true = predictions['diabetes']  
y_pred = predictions['prediction_label']  

accuracy = accuracy_score(y_true, y_pred)

with open('Results/metrics.txt', 'w') as f:
    f.write(f"Acuracia do modelo: {round(accuracy, 3)}\n")
