import streamlit as st
from PIL import Image

st.markdown("# Paper Shortage Visualizer 🎈")
st.sidebar.markdown("# Main 🎈")

image = Image.open('content/sold-out.jpeg')

st.image(image, caption='Interesting but tricky topic...')