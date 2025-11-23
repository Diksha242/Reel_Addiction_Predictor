import streamlit as st

st.set_page_config(page_title="Streamlit Test", layout="centered")

st.title("✅ Streamlit is Working!")
st.write("If you see this, Streamlit is set up correctly.")

name = st.text_input("Enter your name:")

if st.button("Submit"):
    st.success(f"Hello {name}! Streamlit is running perfectly.")
