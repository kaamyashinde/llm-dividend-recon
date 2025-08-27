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
    from agents.breaks_streamlit_integration import display_breaks_results
    from agents.breaks_resolution_agent import BreaksResolutionAgent
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
    
    # Store headers in session state for use in custom mappings
    st.session_state.nbim_headers = nbim_headers
    st.session_state.custody_headers = custody_headers
    
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
    
    # Initialize custom mappings in session state if not exists
    if "custom_mappings" not in st.session_state:
        st.session_state.custom_mappings = []
    
    # Initialize rejected override mappings in session state
    if "rejected_overrides" not in st.session_state:
        st.session_state.rejected_overrides = {}
    
    # Count accepted mappings (including custom ones)
    accepted_count = sum(1 for decision in st.session_state.user_decisions.values() if decision)
    custom_count = len(st.session_state.custom_mappings)
    
    # Simple table
    if mappings_data.get('mappings'):
        st.markdown("### Review Mappings")

        # Apply All button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Accept All", use_container_width=True):
                for i, mapping in enumerate(mappings_data['mappings']):
                    st.session_state.user_decisions[i] = True
                st.rerun()
        with col2:
            if st.button("❌ Reject All", use_container_width=True):
                for i, mapping in enumerate(mappings_data['mappings']):
                    st.session_state.user_decisions[i] = False
                st.rerun()
        st.markdown("---")

        # Prepare editable table data
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

        display_df = pd.DataFrame(display_data)

    # Use SelectboxColumn for 'Custodian' in st.data_editor (Notion-style select)
    custody_headers = getattr(st.session_state, 'custody_headers', [])

    # Track which custodian headers are already accepted
    accepted_custodian_headers = set()
    for i, row in display_df.iterrows():
        if row['Accepted']:
            accepted_custodian_headers.add(row['Custodian'])

    # Use st.data_editor with a single SelectboxColumn for 'Custodian'
    column_config = {
        "Accepted": st.column_config.CheckboxColumn(
            "Accepted",
            help="Check to accept this mapping",
            default=False,
        ),
        "Sr. Nr": st.column_config.NumberColumn("Sr. Nr", disabled=True),
        "NBIM": st.column_config.TextColumn("NBIM", disabled=True),
        "Custodian": st.column_config.SelectboxColumn(
            "Custodian",
            options=custody_headers,
            help="Select the custodian header for this mapping"
        ),
        "Confidence": st.column_config.TextColumn("Confidence", disabled=True),
        "Status": st.column_config.TextColumn("Status", disabled=True)
    }

    edited_df = st.data_editor(
        display_df,
        column_config=column_config,
        disabled=["Sr. Nr", "NBIM", "Confidence", "Status"],
        hide_index=True,
        use_container_width=True
    )

    # Update session state and custom mappings based on changes
    rerun_needed = False
    for i, row in edited_df.iterrows():
        new_decision = row['Accepted']
        orig_custodian = mappings_data['mappings'][i]['target_header']
        new_custodian = row['Custodian']

        # If mapping is accepted and custodian header was changed, treat as custom mapping
        if new_decision and new_custodian != orig_custodian:
            already_custom = any(
                m['source_header'] == row['NBIM'] and m['target_header'] == new_custodian
                for m in st.session_state.custom_mappings
            )
            if not already_custom:
                custom_mapping = {
                    'source_header': row['NBIM'],
                    'target_header': new_custodian,
                    'confidence': 1.0,
                    'reasoning': f'User override of AI suggestion: {orig_custodian} → {new_custodian}',
                    'is_custom': True,
                    'original_ai_suggestion': orig_custodian
                }
                st.session_state.custom_mappings.append(custom_mapping)
                rerun_needed = True
        if st.session_state.user_decisions.get(i, None) != new_decision:
            st.session_state.user_decisions[i] = new_decision
            rerun_needed = True

    # Rerun if any changes
    if not edited_df.equals(display_df) or rerun_needed:
        st.rerun()
    
    # Action buttons
    total_mappings = accepted_count + custom_count
    if total_mappings > 0:
        st.markdown("### Actions")
        
        # Show summary of mappings
        if accepted_count > 0 and custom_count > 0:
            st.info(f"📊 Total: {total_mappings} mappings ({accepted_count} AI accepted + {custom_count} custom)")
        elif accepted_count > 0:
            st.info(f"📊 Total: {accepted_count} AI mappings accepted")
        elif custom_count > 0:
            st.info(f"📊 Total: {custom_count} custom mappings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Start Over", use_container_width=True):
                # Clear all mapping-related state
                st.session_state.user_decisions = {}
                st.session_state.custom_mappings = []
                st.session_state.rejected_overrides = {}
                st.session_state.mapping_result = None
                st.session_state.mappings_applied = False
                st.session_state.effective_mappings = None
                
                # Clear breaks and resolution results
                if "breaks_result" in st.session_state:
                    del st.session_state.breaks_result
                if "resolution_result" in st.session_state:
                    del st.session_state.resolution_result
                if "fix_decisions" in st.session_state:
                    del st.session_state.fix_decisions
                    
                # Clear any analysis flags
                if "breaks_analysis_requested" in st.session_state:
                    del st.session_state.breaks_analysis_requested
                if "resolution_requested" in st.session_state:
                    del st.session_state.resolution_requested
                    
                st.rerun()
        
        with col2:
            if st.button(f"📥 Export {total_mappings} Mappings", use_container_width=True):
                export_accepted_mappings_as_csv(mappings_data)
        
        with col3:
            if st.button("✅ Apply All Mappings", type="primary", use_container_width=True):
                # Build effective mappings only from accepted rows and custom mappings
                effective_mappings = {}
                try:
                    for i, row in edited_df.iterrows():
                        if row.get('Accepted'):
                            src = row.get('NBIM')
                            tgt = row.get('Custodian')
                            if src and tgt:
                                effective_mappings[src] = tgt
                    for cm in st.session_state.custom_mappings:
                        src = cm.get('source_header')
                        tgt = cm.get('target_header')
                        if src and tgt:
                            effective_mappings[src] = tgt
                    st.session_state.effective_mappings = effective_mappings if effective_mappings else None
                    st.session_state.mappings_applied = bool(effective_mappings)
                    if st.session_state.mappings_applied:
                        st.success(f"Applied {len(effective_mappings)} mappings! 🎉")
                    else:
                        st.warning("No accepted mappings to apply. Please accept or add custom mappings.")
                except Exception as e:
                    st.error(f"Failed to apply mappings: {e}")
    
    # Show unmapped headers with ability to add custom mappings
    unmapped = mappings_data.get('unmapped_headers', [])
    if unmapped:
        show_unmapped_headers_section(unmapped, mappings_data)
    
    # Show custom mappings section
    show_custom_mappings_section(mappings_data)
    
    # Remove rejected mappings override section (all overrides now in main table)
    
    # Show manual mapping builder for advanced users
    show_manual_mapping_builder()


def show_unmapped_headers_section(unmapped: list, mappings_data: dict):
    """Show unmapped headers with ability to add custom mappings."""
    
    with st.expander(f"⚠️ {len(unmapped)} headers couldn't be mapped - Add Custom Mappings", expanded=False):
        st.write("**Unmapped Headers:**")
        for header in unmapped:
            st.write(f"• {header}")
        
        st.markdown("---")
        st.write("**Add Custom Mapping:**")
        
        # Get available headers from session state if they exist
        nbim_headers = getattr(st.session_state, 'nbim_headers', [])
        custody_headers = getattr(st.session_state, 'custody_headers', [])
        
        # Get all unmapped NBIM headers (including those not in unmapped list)
        mapped_sources = [m['source_header'] for m in mappings_data.get('mappings', [])] + \
                        [m['source_header'] for m in st.session_state.custom_mappings]
        available_nbim = [h for h in nbim_headers if h not in mapped_sources]
        
        # Show more comprehensive options
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            source_header = st.selectbox(
                "Source Header (NBIM)",
                options=[""] + available_nbim,
                key="custom_source_header",
                help="Select any NBIM header that hasn't been mapped yet"
            )
        
        with col2:
            # Allow both selection and custom input for target
            target_option = st.radio(
                "Target Header Method:",
                ["Select from Custodian", "Enter Custom"],
                key="target_method",
                horizontal=True
            )
            
            if target_option == "Select from Custodian":
                target_header = st.selectbox(
                    "Custodian Header",
                    options=[""] + custody_headers,
                    key="custom_target_select"
                )
            else:
                target_header = st.text_input(
                    "Custom Target Header",
                    placeholder="Enter any header name",
                    key="custom_target_input"
                )
        
        with col3:
            st.write("")  # Space for alignment
            st.write("")  # Space for alignment
            if st.button("➕ Add", key="add_custom", use_container_width=True):
                if source_header and target_header:
                    # Check for duplicates
                    existing_custom = any(
                        m['source_header'] == source_header for m in st.session_state.custom_mappings
                    )
                    if existing_custom:
                        st.error(f"Mapping for '{source_header}' already exists in custom mappings")
                    else:
                        # Add to custom mappings
                        custom_mapping = {
                            'source_header': source_header,
                            'target_header': target_header,
                            'confidence': 1.0,  # User-defined mapping has full confidence
                            'reasoning': 'User-defined custom mapping',
                            'is_custom': True
                        }
                        st.session_state.custom_mappings.append(custom_mapping)
                        st.success(f"✅ Added: {source_header} → {target_header}")
                        st.rerun()
                else:
                    st.error("Please select/enter both source and target headers")


def show_custom_mappings_section(mappings_data: dict):
    """Show and manage custom user-defined mappings."""
    
    if st.session_state.custom_mappings:
        with st.expander(f"🎯 {len(st.session_state.custom_mappings)} Custom Mappings", expanded=True):
            
            # Display custom mappings in a table format
            custom_data = []
            for i, mapping in enumerate(st.session_state.custom_mappings):
                custom_data.append({
                    'Nr': i + 1,
                    'NBIM': mapping['source_header'],
                    'Custodian': mapping['target_header'],
                    'Type': 'Custom',
                    'Remove': False
                })
            
            custom_df = pd.DataFrame(custom_data)
            
            edited_custom_df = st.data_editor(
                custom_df,
                column_config={
                    "Remove": st.column_config.CheckboxColumn(
                        "Remove",
                        help="Check to remove this custom mapping",
                        default=False,
                    ),
                    "Nr": st.column_config.NumberColumn("Nr", disabled=True),
                    "NBIM": st.column_config.TextColumn("NBIM", disabled=True),
                    "Custodian": st.column_config.TextColumn("Custodian", disabled=True),
                    "Type": st.column_config.TextColumn("Type", disabled=True),
                },
                disabled=["Nr", "NBIM", "Custodian", "Type"],
                hide_index=True,
                use_container_width=True,
                key="custom_mappings_editor"
            )
            
            # Handle removals
            removals = []
            for i, row in edited_custom_df.iterrows():
                if row['Remove']:
                    removals.append(i)
            
            if removals:
                # Remove selected mappings
                for index in sorted(removals, reverse=True):
                    removed_mapping = st.session_state.custom_mappings.pop(index)
                    st.success(f"Removed: {removed_mapping['source_header']} → {removed_mapping['target_header']}")
                st.rerun()


def show_rejected_mappings_override_section(mappings_data: dict):
    """Allow users to override rejected AI mappings."""
    
    # Get rejected mappings (those where user_decisions is False)
    rejected_mappings = []
    for i, mapping in enumerate(mappings_data.get('mappings', [])):
        if not st.session_state.user_decisions.get(i, False):  # If not accepted
            rejected_mappings.append((i, mapping))
    
    if rejected_mappings:
        with st.expander(f"🔄 {len(rejected_mappings)} Rejected Mappings - Override if Needed", expanded=False):
            st.write("Sometimes the AI gets it wrong. You can override rejected mappings and create custom versions:")
            
            for original_idx, mapping in rejected_mappings:
                col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
                
                with col1:
                    st.write(f"**{mapping['source_header']}**")
                    st.caption(f"Confidence: {mapping['confidence']:.0%}")
                
                with col2:
                    st.write("→")
                
                with col3:
                    # Allow editing the target header
                    new_target = st.text_input(
                        "Override target:",
                        value=mapping['target_header'],
                        key=f"override_target_{original_idx}",
                        help="Edit the target header if the AI got it wrong"
                    )
                
                with col4:
                    if st.button("✅ Override", key=f"override_btn_{original_idx}"):
                        # Create a custom mapping based on the override
                        override_mapping = {
                            'source_header': mapping['source_header'],
                            'target_header': new_target,
                            'confidence': 1.0,  # User override has full confidence
                            'reasoning': f'User override of AI suggestion: {mapping["target_header"]} → {new_target}',
                            'is_custom': True,
                            'original_ai_suggestion': mapping['target_header']
                        }
                        st.session_state.custom_mappings.append(override_mapping)
                        st.success(f"✅ Override added: {mapping['source_header']} → {new_target}")
                        st.rerun()
                
                st.markdown("---")


def show_manual_mapping_builder():
    """Advanced manual mapping builder for users who want full control."""
    
    # Only show if we have headers available
    if not hasattr(st.session_state, 'nbim_headers') or not hasattr(st.session_state, 'custody_headers'):
        return
    
    with st.expander("🔧 Advanced: Manual Mapping Builder", expanded=False):
        st.write("**Build mappings from scratch or add any missing mappings.**")
        
        nbim_headers = st.session_state.nbim_headers
        custody_headers = st.session_state.custody_headers
        
        # Show unmapped headers from both sides
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Available NBIM Headers:**")
            # Get already mapped NBIM headers
            mapped_nbim = set()
            if hasattr(st.session_state, 'mapping_result') and st.session_state.mapping_result:
                for i, mapping in enumerate(st.session_state.mapping_result.get('mappings', [])):
                    if st.session_state.user_decisions.get(i, False):
                        mapped_nbim.add(mapping['source_header'])
            
            # Add custom mappings
            for mapping in st.session_state.custom_mappings:
                mapped_nbim.add(mapping['source_header'])
            
            unmapped_nbim = [h for h in nbim_headers if h not in mapped_nbim]
            
            if unmapped_nbim:
                for header in unmapped_nbim:
                    st.write(f"• {header}")
            else:
                st.write("✅ All NBIM headers are mapped")
        
        with col2:
            st.write("**Available Custodian Headers:**")
            for header in custody_headers:
                st.write(f"• {header}")
        
        st.markdown("---")
        
        # Quick mapping interface
        st.write("**Quick Add Mapping:**")
        col1, col2, col3 = st.columns([3, 3, 2])
        
        with col1:
            manual_source = st.selectbox(
                "NBIM Header",
                options=[""] + nbim_headers,
                key="manual_source_header"
            )
        
        with col2:
            manual_target = st.selectbox(
                "Custodian Header", 
                options=[""] + custody_headers,
                key="manual_target_header"
            )
        
        with col3:
            st.write("")  # Spacing
            if st.button("Add Mapping", key="manual_add", use_container_width=True):
                if manual_source and manual_target:
                    # Check for duplicates
                    existing = any(
                        m['source_header'] == manual_source for m in st.session_state.custom_mappings
                    )
                    if existing:
                        st.error(f"'{manual_source}' is already mapped")
                    else:
                        custom_mapping = {
                            'source_header': manual_source,
                            'target_header': manual_target,
                            'confidence': 1.0,
                            'reasoning': 'Manual mapping via advanced builder',
                            'is_custom': True
                        }
                        st.session_state.custom_mappings.append(custom_mapping)
                        st.success(f"✅ Added: {manual_source} → {manual_target}")
                        st.rerun()
                else:
                    st.error("Please select both headers")


def export_accepted_mappings_as_csv(mappings_data: dict):
    """Export mappings as downloadable CSV including custom mappings."""
    
    # Combine AI mappings and custom mappings
    all_mappings = []
    
    # Add accepted AI mappings
    if mappings_data.get('mappings'):
        for i, mapping in enumerate(mappings_data['mappings']):
            if st.session_state.user_decisions.get(i, False):  # If accepted
                all_mappings.append({
                    "source_header": mapping['source_header'],
                    "target_header": mapping['target_header'],
                    "confidence": mapping['confidence'],
                    "reasoning": mapping['reasoning'],
                    "type": "AI Generated"
                })
    
    # Add custom mappings
    for mapping in st.session_state.custom_mappings:
        all_mappings.append({
            "source_header": mapping['source_header'],
            "target_header": mapping['target_header'], 
            "confidence": mapping['confidence'],
            "reasoning": mapping['reasoning'],
            "type": "User Custom"
        })
    
    if not all_mappings:
        st.warning("No mappings to export.")
        return
    
    export_df = pd.DataFrame(all_mappings)
    csv_data = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download All Mappings CSV",
        data=csv_data,
        file_name="complete_header_mappings.csv",
        mime="text/csv",
        use_container_width=True
    )


st.set_page_config(layout="wide")
st.title("LLM Dividend Reconciliation & Resolution Application")

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
        
        # Build mappings for break identification
        mappings = None
        if hasattr(st.session_state, 'mapping_result') and st.session_state.mapping_result:
            mapping_data = st.session_state.mapping_result
            mappings = {}
            ai_mappings = mapping_data.get('mappings', []) or []

            # Use accepted mappings if user decisions exist, else fall back to AI suggestions
            user_decisions = getattr(st.session_state, 'user_decisions', {}) or {}
            if isinstance(user_decisions, dict) and user_decisions:
                for i, m in enumerate(ai_mappings):
                    if user_decisions.get(i, False):
                        mappings[m['source_header']] = m['target_header']
            else:
                # No explicit accept/reject yet; use AI suggestions to enable analysis
                for m in ai_mappings:
                    mappings[m['source_header']] = m['target_header']

            # Overlay user custom mappings if any
            custom_mappings = getattr(st.session_state, 'custom_mappings', []) or []
            for cm in custom_mappings:
                src = cm.get('source_header')
                tgt = cm.get('target_header')
                if src and tgt:
                    mappings[src] = tgt
            # Ensure None if we ended up with empty dict
            if not mappings:
                mappings = None
        
        # Only run reconciliation when user applies mappings AND breaks haven't been analyzed yet
        if AGENT_AVAILABLE and getattr(st.session_state, 'mappings_applied', False):
            effective_mappings = getattr(st.session_state, 'effective_mappings', None)
            # Add guard: only run if we don't already have breaks results
            if effective_mappings and not hasattr(st.session_state, 'breaks_result'):
                try:
                    from agents.breaks_identifier_agent import BreaksIdentifierAgent
                    agent = BreaksIdentifierAgent()
                    # Restrict payload to only mapped columns to avoid spurious comparisons
                    nbim_cols = list(effective_mappings.keys())
                    custody_cols = list(effective_mappings.values())
                    nbim_slim = nbim_df[ [c for c in nbim_cols if c in nbim_df.columns] ].copy()
                    custody_slim = custody_df[ [c for c in custody_cols if c in custody_df.columns] ].copy()
                    with st.spinner("🤖 AI is identifying breaks using the applied mappings..."):
                        result = asyncio.run(agent.identify_breaks(
                            nbim_data=nbim_slim.to_dict('records'),
                            custody_data=custody_slim.to_dict('records'),
                            mappings=effective_mappings,
                            additional_context=None,
                            timeout=180.0
                        ))
                    if agent.validate_breaks(result):
                        st.success("✅ Breaks analysis completed!")
                        # Store the result to prevent re-running
                        st.session_state.breaks_result = result.response.structured_data
                        # Ensure resolution_result session state is initialized
                        if "resolution_result" not in st.session_state:
                            st.session_state.resolution_result = None
                        display_breaks_results(result.response.structured_data)
                    else:
                        err = result.error if result and result.error else 'Validation failed'
                        st.error(f"❌ Breaks analysis failed: {err}")
                except Exception as e:
                    st.error(f"🚨 Error running breaks identification: {e}")
            elif effective_mappings and hasattr(st.session_state, 'breaks_result'):
                # Display existing results without re-running analysis
                if "resolution_result" not in st.session_state:
                    st.session_state.resolution_result = None
                display_breaks_results(st.session_state.breaks_result)
            else:
                st.info("Apply mappings to run reconciliation analysis.")
        else:
            st.info("Apply mappings to run reconciliation analysis.")
        
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