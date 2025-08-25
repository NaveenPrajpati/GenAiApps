import streamlit as st
from basicApps.BlogToPodcast.app import blogToPodcast
from basicApps.DataAnalyst.app import dataAnalyst
from basicApps.DataVisulalize.app import dataVisualize
from basicApps.MedicalImageFinder.app import medicalImageDiagnose
from basicApps.MemeGenerator.app import generateMeme
from basicApps.BreakupRecovery.app import breakupRecovery

option = st.sidebar.selectbox(
    "Basic Agents:",
    (
        "Blog to podcast",
        "Data Analyst",
        "Data Visualize",
        "Medical Image Diagnose",
        "Generate Meme",
        "Breakup Recovery",
    ),
)

if option == "Blog to podcast":
    blogToPodcast()
elif option == "Data Analyst":
    dataAnalyst()
elif option == "Data Visualize":
    dataVisualize()
elif option == "Medical Image Diagnose":
    medicalImageDiagnose()
elif option == "Generate Meme":
    generateMeme()
elif option == "Breakup Recovery":
    breakupRecovery()
