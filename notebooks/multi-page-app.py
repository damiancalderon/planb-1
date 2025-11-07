import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
# Set the page title
st.set_page_config(page_title="Multi-Page Dashboard", layout="wide")
# Define the page options for the sidebar
page = st.sidebar.selectbox("Select an option", ["Home", "Seaborn Charts",
"Plotly Charts"])
st.title(f"Welcome to the {page} page!")
# Display content based on the selected page
if page == "Home":
    st.write("This is the home page. Select a page from the sidebar to view the charts. ")
if page == "Seaborn Charts":
    st.header("Seaborn Charts")
if page == "Plotly Charts":
    st.header("Plotly Charts")