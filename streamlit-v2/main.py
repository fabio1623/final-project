import streamlit as st
import pandas as pd
from PIL import Image

def main():
    st.title("Welcome to the Home Page!")

    image = Image.open('content/sold-out.jpeg')
    st.image(image, caption='Interesting but tricky topic...')

if __name__ == "__main__":
    main()
