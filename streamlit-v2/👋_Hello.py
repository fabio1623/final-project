import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to the Paper Shortage Visualizer! 👋")

image = Image.open('content/sold-out.jpeg')
st.image(image, caption='Interesting but tricky topic...')