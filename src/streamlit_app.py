from typing import Dict
import logging

import streamlit as st
from transformers import AutoTokenizer

from storyteller import StoryTeller
import config


LOGGER = logging.getLogger(__name__)
__version__ = "0.0.1"

@st.cache_data
def load_eos_token_id(model_name: int = config.model_name_for_eos_token_id) -> int:
    LOGGER.info(f"Loading tokenizer from model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.eos_token_id


with st.sidebar:
    st.title("The Lord of the Rings Storyteller")
    st.header("Write your own story!")
    st.write("This app enables you to complete a 'The Lord of the Rings' story based on your inital text.")

    st.header("Generation parameters")
    st.slider("Temperature", min_value=0.0, max_value=1.0, value=config.temperature, step=0.1, key="temperature")
    st.slider("Max new tokens", min_value=0, max_value=1000, value=config.max_new_tokens, step=10, key="max_new_tokens")
    st.slider("Repetition penalty", min_value=0.0, max_value=5.0, value=config.repetition_penalty, step=0.2, key="repetition_penalty")
    st.checkbox("Do sample", value=config.do_sample, key="do_sample")


def write(text: str) -> str:
    storyteller = StoryTeller()
    eos_token_id = load_eos_token_id()
    parameters = {
        "max_new_tokens": st.session_state.max_new_tokens,
        "do_sample": st.session_state.do_sample,
        "temperature": st.session_state.temperature,
        "early_stopping": config.early_stopping,
        "repetition_penalty": st.session_state.repetition_penalty,
        "forced_eos_token_id": eos_token_id
    }
    LOGGER.info(f"Parameters: {parameters}")
    output = storyteller({"inputs": text, "parameters": parameters})
    LOGGER.info(f"Writing finished. The result is: {output}")
    return output["generated_text"]


st.text_input("Start your story here", key="text_input")
if st.button("Generate"):
    text = st.session_state.text_input
    with st.spinner("Generating story..."):
        if text:
            story = write(text)
            st.write(story)
        else:
            st.warning("Please enter some text to start your story.")

