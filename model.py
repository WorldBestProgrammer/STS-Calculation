import yaml
import torch
import streamlit as st
import transformers

@st.cache_data
def load_model():
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model_path = "./model/" + config["name"]
    model = transformers.AutoModelForSequenceClassification.from_pretrained("./model.bin").to('mps')
    
    return model

