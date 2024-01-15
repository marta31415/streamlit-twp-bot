import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

table = pd.read_csv('data.csv')[['time','name','value','variable']]

def getAnswer(table,question):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    
    encoding = tokenizer(table, question, return_tensors="pt")
    outputs = model.generate(**encoding)
    predicted_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return predicted_answer

def main():
    st.set_page_config(
        page_title="The Weather Partner BOT", page_icon=":robot_face:")

    st.header("DEMO :robot_face: The Weather Partner BOT :robot_face:")
    
    st.write("The dates available are from 2023-01-01 to 2023-01-03 included and the stations available are Cano (CNO), Corozal Oeste (CZL), Dos Bocas (DBK), Escandalosa (ESC), Balboa FAA (Federal Aviation Authority) (FAA). You can ask questions about precipitation or temperature.")
    message = st.text_area("Ask a question...")
    
    options = [
        "",
        "What was the total precipitation in Cano on 2023-01-01 00:00:00?",
        "What was the temperature in DBK on 2023-01-01 00:00:00?",
        "Where was the maximum precipitation on 2023-01-02 00:00:00?",
        "Where was the maximum temperature on 2023-01-01 00:00:00?",
        "What was the total precipitation in Balboa FAA on the 2023-01-02 00:00:00?"
    ]

    text_input_with_dropdown = st.selectbox("... or choose an example question:", options)

    if message or text_input_with_dropdown!="":
        st.write("Generating the answer...")
        if message:
            result = getAnswer(table,message)
        else:
            result = getAnswer(table,text_input_with_dropdown)

        st.info(result)

        st.text("You can check the answer at the table below.")
        st.table(table.rename(columns={
            'time':'Date',
            'name':'Station',
            'variable':'Variable',
            'value':'Value',
        }))


if __name__ == '__main__':
    main()