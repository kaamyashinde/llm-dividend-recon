import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("LLM Dividend Reconcilation Application")

# File upload system
st.markdown("## Upload Files")

# Step 1: Upload NBIM file
nbim_file = st.file_uploader("Upload NBIM file", type=['csv'], key="nbim")

# Step 2: Upload Custody file (only show if NBIM file is uploaded)
if nbim_file is not None:
    custody_file = st.file_uploader("Upload Custody file", type=['csv'], key="custody")
    
    # Process files when both are uploaded
    if custody_file is not None:
        # Read the uploaded files into dataframes
        nbim_df = pd.read_csv(nbim_file, sep=';')
        custody_df = pd.read_csv(custody_file, sep=';')
        
        st.success("Both files uploaded successfully!")
        
        # Display tables side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("## NBIM File")
            st.write(nbim_df.T)
        
        with col2:
            st.markdown("## Custody File")
            st.write(custody_df.T)
else:
    st.info("Please upload the NBIM file first to continue.")