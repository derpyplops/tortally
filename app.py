import streamlit as st
from run import run
import asyncio

def main():
    st.title("Tortally")
    lcase = st.text_input("Enter a summary of a case:")

    if st.button("Submit"):
        with st.spinner('Calculating...'):
            result = asyncio.run(run(lcase))
        st.write(result)

if __name__ == '__main__':
    main()
