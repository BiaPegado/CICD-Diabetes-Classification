import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

modelo = load_model('model/diabetes_model')

def predict_diabetes(age, gender, bmi, hypertension, heart_disease, smoking_history, hba1c_level, blood_glucose_level):

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
    
    predictions = predict_model(modelo, data=input_data)
    resultado = predictions['prediction_label'][0] 
    
    return "Diabético" if resultado == 1 else "Não Diabético"

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("""
    # Previsão de Diabetes
    Insira as informações do paciente para prever a presença de diabetes. Use valores precisos para obter resultados confiáveis.
    """)

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Idade", value=30, precision=0)
            gender = gr.Radio(label="Gênero", choices=["Masculino", "Feminino"], value="Masculino")
            bmi = gr.Number(label="Índice de Massa Corporal (BMI)", value=25.0)

        with gr.Column():
            hypertension = gr.Checkbox(label="Hipertensão")
            heart_disease = gr.Checkbox(label="Doença Cardíaca")
            smoking_history = gr.Radio(
                label="Histórico de Fumo",
                choices=["Nunca", "Ocasional", "Frequente"],
                value="Nunca"
            )

    with gr.Row():
        hba1c_level = gr.Number(label="Nível de HBA1c (%)", value=5.5)
        blood_glucose_level = gr.Number(label="Nível de Glicose no Sangue (mg/dL)", value=100)

    output = gr.Textbox(label="Resultado da Previsão", lines=2)

    gr.Button("Prever").click(
        predict_diabetes,
        inputs=[
            age, gender, bmi, hypertension, heart_disease,
            smoking_history, hba1c_level, blood_glucose_level
        ],
        outputs=output
    )

interface.launch(share=True)
