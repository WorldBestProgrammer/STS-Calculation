import streamlit as st
from model import load_model
from data import preprocessing
import transformers
import yaml

def main():
    
    st.title("STS 측정")

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"], max_length=160)

    model = load_model()

    with st.form(key="입력 form"):
      sentence1 = st.text_input("sentence1")
      sentence2 = st.text_input("sentence2")
      submitted = st.form_submit_button("Confirm")
      
      if submitted:
          input = preprocessing(tokenizer, sentence1, sentence2).to('mps')
          result = model(input)['logits'].item() * 20
          result = str(int(round(result, 1)))
          st.write(result + "%")
          print(result)
          
    

main()