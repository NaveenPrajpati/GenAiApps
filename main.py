import streamlit as st
from basicApps.BlogToPodcast.app import blogToPodcast
from basicApps.DataAnalyst.app import dataAnalyst
from basicApps.DataVisulalize.app import dataVisualize
from basicApps.MedicalImageFinder.app import medicalImageDiagnose
from basicApps.MemeGenerator.app import generateMeme
from basicApps.BreakupRecovery.app import breakupRecovery
from basicApps.WebScraper.app import webScraper

option = st.sidebar.selectbox(
    "Basic Agents:",
    (
        "Blog to podcast",
        "Data Analyst",
        "Data Visualize",
        "Medical Image Diagnose",
        "Generate Meme",
        "Breakup Recovery",
        "Web Scraper",
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
elif option == "Web Scraper":
    webScraper()
