import streamlit as st
import pandas as pd
import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path for imports
sys.path.append(os.path.dirname(__file__))

# Import our custom agents and utils
try:
    from agents.mappings_agent import MappingsAgent
    from utils import logger
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    st.error(f"Agent import failed: {e}. Please ensure utils and agents folders are set up correctly.")

# Helper functions for AI-powered mapping
def add_header_mapping_section(nbim_df: pd.DataFrame, custody_df: pd.DataFrame):
    """Add the AI-powered header mapping section to the app."""
    
    if not AGENT_AVAILABLE:
        st.warning("⚠️ AI mapping not available. Please set up the agents and utils packages.")
        return
    
    st.markdown("## 🤖 AI Header Mapping")
    
    nbim_headers = list(nbim_df.columns)
    custody_headers = list(custody_df.columns)
    
    # Check if mapping has been done
    if "mapping_result" not in st.session_state:
        st.session_state.mapping_result = None
    
    # Create mapping button - centered and prominent
    if st.session_state.mapping_result is None:
        st.markdown("AI can automatically map headers between your files for easier reconciliation.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🧠 Generate Mappings", type="primary", use_container_width=True):
                run_ai_mapping(nbim_headers, custody_headers)
    
    # Display results if available
    if st.session_state.mapping_result:
        display_mapping_results(st.session_state.mapping_result)


@st.cache_data(ttl=300, show_spinner="🤖 AI is analyzing headers...")
def run_ai_mapping(nbim_headers: list, custody_headers: list):
    """Run the AI mapping agent and cache results."""
    
    try:
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            st.error("🔑 OpenAI API key not found! Please add OPENAI_API_KEY to your .env file.")
            return None
        
        # Create context for the AI
        context = f"""
        Financial data reconciliation between NBIM (Norwegian Bank Investment Management) 
        and Custody bank files. 
        
        NBIM file has {len(nbim_headers)} columns: {', '.join(nbim_headers)}
        Custody file has {len(custody_headers)} columns: {', '.join(custody_headers)}
        
        This is dividend reconciliation data containing securities information, 
        quantities, prices, and market values.
        """
        
        # Create and run the agent
        agent = MappingsAgent()
        
        # Run async function in Streamlit
        with st.spinner("🤖 AI is creating header mappings..."):
            result = asyncio.run(agent.create_mappings(
                source_headers=nbim_headers,
                target_headers=custody_headers,
                context=context.strip(),
                timeout=90.0
            ))
        
        # Validate results
        if agent.validate_mappings(result):
            st.session_state.mapping_result = result.response.structured_data
            st.success("✅ AI mapping completed successfully!")
            return result.response.structured_data
        else:
            error_msg = str(result.error) if result.error else "Unknown validation error"
            st.error(f"❌ AI mapping failed: {error_msg}")
            return None
            
    except Exception as e:
        st.error(f"🚨 Error running AI mapping: {str(e)}")
        logger.error(f"Streamlit AI mapping error: {e}")
        return None


def display_mapping_results(mappings_data: dict):
    """Display the AI mapping results as an unformatted table (DataFrame-like)."""
    
    # Initialize user decisions in session state if not exists
    if "user_decisions" not in st.session_state:
        st.session_state.user_decisions = {}
    
    # Count accepted mappings
    accepted_count = sum(1 for decision in st.session_state.user_decisions.values() if decision)
    
    # Simple table
    if mappings_data.get('mappings'):
        st.markdown("### Review Mappings")
        
        # Apply All button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Accept All", use_container_width=True):
                # Accept all mappings
                for i, mapping in enumerate(mappings_data['mappings']):
                    st.session_state.user_decisions[i] = True
                st.rerun()
        
        with col2:
            if st.button("❌ Reject All", use_container_width=True):
                # Reject all mappings (default state)
                for i, mapping in enumerate(mappings_data['mappings']):
                    st.session_state.user_decisions[i] = False
                st.rerun()
        
        st.markdown("---")
        
        # Create DataFrame for display with interactive checkbox column
        display_data = []
        
        for i, mapping in enumerate(mappings_data['mappings']):
            current_decision = st.session_state.user_decisions.get(i, False)
            display_data.append({
                'Sr. Nr': i + 1,
                'NBIM': mapping['source_header'],
                'Custodian': mapping['target_header'],
                'Confidence': f"{mapping['confidence']:.0%}",
                'Status': "Accepted" if current_decision else "Rejected",
                'Accepted': current_decision
            })
        
        # Create and display the DataFrame with interactive checkbox column
        display_df = pd.DataFrame(display_data)
        
        edited_df = st.data_editor(
            display_df,
            column_config={
                "Accepted": st.column_config.CheckboxColumn(
                    "Accepted",
                    help="Check to accept this mapping",
                    default=False,
                ),
                "Sr. Nr": st.column_config.NumberColumn(
                    "Sr. Nr",
                    disabled=True,
                ),
                "NBIM": st.column_config.TextColumn(
                    "NBIM",
                    disabled=True,
                ),
                "Custodian": st.column_config.TextColumn(
                    "Custodian", 
                    disabled=True,
                ),
                "Confidence": st.column_config.TextColumn(
                    "Confidence",
                    disabled=True,
                ),
                "Status": st.column_config.TextColumn(
                    "Status",
                    disabled=True,
                )
            },
            disabled=["Sr. Nr", "NBIM", "Custodian", "Confidence", "Status"],
            hide_index=True,
            use_container_width=True
        )
        
        # Update session state based on checkbox changes
        for i, row in edited_df.iterrows():
            new_decision = row['Accepted']
            old_decision = st.session_state.user_decisions.get(i, False)
            
            if new_decision != old_decision:
                st.session_state.user_decisions[i] = new_decision
                # Update the status in the dataframe
                edited_df.at[i, 'Status'] = "Accepted" if new_decision else "Rejected"
        
        # Check if any changes were made and rerun if needed
        if not edited_df.equals(display_df):
            st.rerun()
    
    # Action buttons
    if accepted_count > 0:
        st.markdown("### Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.user_decisions = {}
                st.session_state.mapping_result = None
                st.rerun()
        
        with col2:
            if st.button(f"📥 Export {accepted_count} Accepted", use_container_width=True):
                export_accepted_mappings_as_csv(mappings_data)
        
        with col3:
            if st.button("✅ Apply Mappings", type="primary", use_container_width=True):
                st.success(f"Applied {accepted_count} mappings! 🎉")
    
    # Show unmapped headers if any
    unmapped = mappings_data.get('unmapped_headers', [])
    if unmapped:
        with st.expander(f"⚠️ {len(unmapped)} headers couldn't be mapped"):
            for header in unmapped:
                st.write(f"• {header}")


def export_accepted_mappings_as_csv(mappings_data: dict):
    """Export mappings as downloadable CSV."""
    
    if not mappings_data.get('mappings'):
        st.warning("No mappings to export.")
        return
    
    # Create export dataframe
    export_data = []
    for mapping in mappings_data['mappings']:
        export_data.append({
            "source_header": mapping['source_header'],
            "target_header": mapping['target_header'],
            "confidence": mapping['confidence'],
            "reasoning": mapping['reasoning']
        })
    
    export_df = pd.DataFrame(export_data)
    csv_data = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Mappings CSV",
        data=csv_data,
        file_name="ai_header_mappings.csv",
        mime="text/csv",
        use_container_width=True
    )


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
        
        # Add AI-powered header mapping section
        add_header_mapping_section(nbim_df, custody_df)
        
        # Display tables side by side
        st.markdown("## Data Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### NBIM File")
            st.dataframe(nbim_df, use_container_width=True)
        
        with col2:
            st.markdown("### Custody File") 
            st.dataframe(custody_df, use_container_width=True)
else:
    st.info("Please upload the NBIM file first to continue.")