import streamlit as st

st.title("My First App")

user_name = st.text_input("Enter your name : ")



if user_name :
    st.write(f"Welcome : {user_name}" )

if st.button("Click Me!"):
    st.write(f"Button Clickedv" )
