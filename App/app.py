import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

# Carregar o modelo treinado (substitua 'diabetes_model' pelo nome do arquivo do modelo salvo)
modelo = load_model('diabetes_model')

# Função para fazer previsões
def predict_diabetes(age, gender, bmi, hypertension, heart_disease, smoking_history, hba1c_level, blood_glucose_level):
    # Criar um dataframe com os nomes corretos
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [1 if gender == "Masculino" else 0],  
        'bmi': [bmi],
        'hypertension': [1 if hypertension else 0],
        'heart_disease': [1 if heart_disease else 0],
        'smoking_history': [smoking_history],
        'HbA1c_level': [hba1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })
    
    # Fazer previsão usando o modelo carregado
    predictions = predict_model(modelo, data=input_data)
    resultado = predictions['prediction_label'][0] 
    
    return "Diabético" if resultado == 1 else "Não Diabético"

# Interface do Gradio
inputs = [
    gr.Number(label="Idade", value=30, precision=0),
    gr.Radio(label="Gênero", choices=["Masculino", "Feminino"], value="Masculino"),
    gr.Number(label="Índice de Massa Corporal (BMI)", value=25.0),
    gr.Checkbox(label="Hipertensão"),
    gr.Checkbox(label="Doença Cardíaca"),
    gr.Radio(label="Histórico de Fumo", choices=["Nunca", "Ocasional", "Frequente"], value="Nunca"),
    gr.Number(label="Nível de HBA1c (%)", value=5.5),
    gr.Number(label="Nível de Glicose no Sangue (mg/dL)", value=100)
]

# Criar a interface
interface = gr.Interface(
    fn=predict_diabetes, 
    inputs=inputs, 
    outputs="text", 
    title="Previsão de Diabetes",
    description="Insira as informações do paciente para prever a presença de diabetes."
)

# Executar a interface
interface.launch(share=True)
