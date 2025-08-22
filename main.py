import streamlit as st
from basicApps.BlogToPodcast.app import blogToPodcast
from basicApps.DataAnalyst.app import dataAnalyst

option = st.sidebar.selectbox(
    "Basic Agents:", ("Blog to podcast", "Data Analyst", "Contact")
)

if option == "Blog to podcast":
    blogToPodcast()
elif option == "Data Analyst":
    dataAnalyst()
elif option == "Contact":
    st.title("Contact Page")
    st.write("Contact us at: hello@example.com")
