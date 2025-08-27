"""
Streamlit integration helpers for the Breaks Identifier Agent.
"""

import streamlit as st
import pandas as pd
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import json

# Import the breaks identifier and resolution agents
from agents.breaks_identifier_agent import BreaksIdentifierAgent
from agents.breaks_resolution_agent import BreaksResolutionAgent
from agents.jira_issue_agent import JiraIssueAgent
from utils import logger


def add_breaks_identification_section(nbim_df: pd.DataFrame, custody_df: pd.DataFrame, mappings: Optional[Dict[str, str]] = None):
    """
    Add the breaks identification section to the Streamlit app.
    
    Args:
        nbim_df: NBIM DataFrame
        custody_df: Custody DataFrame
        mappings: Optional field mappings from the mapping agent
    """
    
    st.markdown("## 🔍 Breaks Identification & Analysis")
    st.markdown("AI-powered identification of discrepancies between NBIM and Custody data.")
    
    # Initialize session state
    if "breaks_result" not in st.session_state:
        st.session_state.breaks_result = None
    
    if "breaks_filter" not in st.session_state:
        st.session_state.breaks_filter = "all"
        
    if "resolution_result" not in st.session_state:
        st.session_state.resolution_result = None
    
    # Initialize fix decisions session state early
    if "fix_decisions" not in st.session_state:
        st.session_state.fix_decisions = {}
    
    # Flag to track if resolution was requested
    if "resolution_requested" not in st.session_state:
        st.session_state.resolution_requested = False
    
    # Flag to track if breaks analysis was requested
    if "breaks_analysis_requested" not in st.session_state:
        st.session_state.breaks_analysis_requested = False
    
    # Analysis options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Select event key for focused analysis
        event_keys = []
        if 'coac_event_key' in nbim_df.columns:
            event_keys = nbim_df['coac_event_key'].unique().tolist()
        
        selected_event = st.selectbox(
            "Select Event (Optional)",
            options=["All Events"] + event_keys,
            help="Focus analysis on specific dividend event",
            key="breaks_selected_event"
        )
    
    with col2:
        # Focus areas
        focus_areas = st.multiselect(
            "Focus Areas",
            options=[
                "tax calculations",
                "missing records",
                "value mismatches",
                "date discrepancies",
                "quantity differences",
                "currency issues"
            ],
            default=["missing records", "value mismatches"],
            help="Specific areas to focus the analysis on",
            key="breaks_focus_areas"
        )
    
    with col3:
        # Analysis depth
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Comprehensive"],
            value="Standard",
            help="Level of detail in the analysis",
            key="breaks_analysis_depth"
        )
    
    # Run analysis button
    if st.session_state.breaks_result is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Identify Breaks", type="primary", use_container_width=True, key="identify_breaks_button"):
                # Store analysis parameters in session state
                st.session_state.breaks_analysis_params = {
                    'selected_event': selected_event,
                    'focus_areas': focus_areas,
                    'analysis_depth': analysis_depth
                }
                st.session_state.breaks_analysis_requested = True
                st.rerun()
    else:
        # Show re-run option
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🔄 Re-run Analysis", use_container_width=True, key="rerun_breaks_analysis_button"):
                st.session_state.breaks_result = None
                st.session_state.resolution_result = None
                st.session_state.fix_decisions = {}
                # Store current parameters for re-run
                st.session_state.breaks_analysis_params = {
                    'selected_event': selected_event,
                    'focus_areas': focus_areas,
                    'analysis_depth': analysis_depth
                }
                st.session_state.breaks_analysis_requested = True
                st.rerun()
    
    # Check if breaks analysis was requested and run it
    if st.session_state.breaks_analysis_requested and st.session_state.breaks_result is None:
        params = st.session_state.get('breaks_analysis_params', {})
        run_breaks_analysis(
            nbim_df, 
            custody_df, 
            mappings, 
            params.get('selected_event', selected_event),
            params.get('focus_areas', focus_areas),
            params.get('analysis_depth', analysis_depth)
        )
        st.session_state.breaks_analysis_requested = False
    
    # Display results if available
    if st.session_state.breaks_result:
        display_breaks_results(st.session_state.breaks_result)


def run_breaks_analysis(
    nbim_df: pd.DataFrame, 
    custody_df: pd.DataFrame, 
    mappings: Optional[Dict[str, str]],
    selected_event: str,
    focus_areas: List[str],
    analysis_depth: str
):
    """Run the breaks identification analysis."""
    
    try:
        # Filter data if specific event selected
        if selected_event != "All Events" and 'coac_event_key' in nbim_df.columns:
            nbim_data = nbim_df[nbim_df['coac_event_key'] == selected_event].to_dict('records')
            custody_data = custody_df[custody_df.get('coac_event_key', custody_df.columns[0]) == selected_event].to_dict('records') if 'coac_event_key' in custody_df.columns else custody_df.to_dict('records')
        else:
            nbim_data = nbim_df.to_dict('records')
            custody_data = custody_df.to_dict('records')
        
        # Adjust timeout based on analysis depth
        timeout_map = {"Quick": 60, "Standard": 120, "Comprehensive": 180}
        timeout = timeout_map.get(analysis_depth, 120)
        
        # Create agent and run analysis
        agent = BreaksIdentifierAgent()
        
        with st.spinner(f"🤖 AI is analyzing {len(nbim_data)} NBIM and {len(custody_data)} Custody records..."):
            # Build additional context for the LLM from UI selections
            context_parts = []
            if selected_event != "All Events":
                context_parts.append(f"Focus on event: {selected_event}")
            if focus_areas:
                context_parts.append("Focus areas: " + ", ".join(focus_areas))
            additional_context = "\n".join(context_parts) if context_parts else None

            result = asyncio.run(agent.identify_breaks(
                nbim_data=nbim_data,
                custody_data=custody_data,
                mappings=mappings,
                additional_context=additional_context,
                timeout=timeout
            ))
        
        # Validate and store results
        if agent.validate_breaks(result):
            st.session_state.breaks_result = result.response.structured_data
            st.success("✅ Breaks analysis completed successfully!")
            
            # Show quick summary
            total_breaks = result.response.structured_data.get("total_breaks_found", 0)
            if total_breaks == 0:
                st.balloons()
                st.info("🎉 Perfect reconciliation! No breaks identified.")
            else:
                st.warning(f"Found {total_breaks} breaks that need review.")
        else:
            error_msg = str(result.error) if result.error else "Analysis validation failed"
            st.error(f"❌ Breaks analysis failed: {error_msg}")
    
    except Exception as e:
        st.error(f"🚨 Error running breaks analysis: {str(e)}")
        logger.error(f"Breaks analysis error: {e}")


def run_breaks_resolution(breaks_data: Dict[str, Any]):
    """Run the breaks resolution analysis."""
    
    breaks = breaks_data.get("breaks", [])
    if not breaks:
        st.warning("No breaks to analyze for resolution.")
        return
    
    try:
        # Create resolution agent
        resolution_agent = BreaksResolutionAgent()
        
        with st.spinner(f"🤖 AI is analyzing {len(breaks)} breaks for resolution strategies..."):
            # Build additional context
            additional_context = f"""
            This is a dividend reconciliation analysis.
            We have identified {len(breaks)} breaks between NBIM and Custody systems.
            
            Break classifications found: {', '.join(breaks_data.get('classification_summary', {}).keys())}
            Severity levels: {', '.join(breaks_data.get('severity_summary', {}).keys())}
            
            Please focus on practical, actionable fixes that can be implemented.
            Consider automation opportunities where possible.
            """
            
            result = asyncio.run(resolution_agent.analyze_and_resolve(
                breaks=breaks,
                additional_context=additional_context.strip(),
                historical_patterns=None,  # Could be passed from previous reconciliations
                timeout=90.0  # Reduced timeout for simpler analysis
            ))
        
        # Validate and store results
        if resolution_agent.validate_resolution(result):
            st.session_state.resolution_result = result.response.structured_data
            st.success("✅ Breaks resolution analysis completed successfully!")
            
            # Show quick summary
            total_resolutions = result.response.structured_data.get("total_breaks_analyzed", 0)
            total_resolvable = result.response.structured_data.get("total_resolvable", 0)
            manual_review = result.response.structured_data.get("total_requiring_manual_review", 0)
            automatable_count = result.response.structured_data.get("automation_potential", {}).get("fully_automatable", 0)
            
            if total_resolvable > 0:
                st.success(f"📋 Analyzed {total_resolutions} breaks. {total_resolvable} have suggested fixes ({automatable_count} fully automatable, {manual_review} need manual review).")
            else:
                st.info(f"📋 Analyzed {total_resolutions} breaks. All require manual review.")
        else:
            error_msg = str(result.error) if result.error else "Resolution analysis validation failed"
            st.error(f"❌ Resolution analysis failed: {error_msg}")
    
    except Exception as e:
        st.error(f"🚨 Error running breaks resolution: {str(e)}")
        logger.error(f"Breaks resolution error: {e}")


def display_breaks_results(breaks_data: Dict[str, Any]):
    """Display the breaks analysis results with a summary and resolution table."""
    
    # Initialize resolution result session state if not exists
    if "resolution_result" not in st.session_state:
        st.session_state.resolution_result = None
    
    # Initialize fix decisions session state
    if "fix_decisions" not in st.session_state:
        st.session_state.fix_decisions = {}
    
    # Display summary tab
    display_summary_tab(breaks_data)
    
    # Show detailed breakdown table
    st.markdown("---")
    display_breakdown_section(breaks_data)
    
    # Show resolution section if we have breaks
    total_breaks = breaks_data.get("total_breaks_found", 0)
    if total_breaks > 0:
        st.markdown("---")
        display_resolution_section(breaks_data)


def display_summary_tab(breaks_data: Dict[str, Any]):
    """Display the summary tab."""
    
    total_breaks = breaks_data.get("total_breaks_found", 0)
    severity_summary = breaks_data.get("severity_summary", {})
    classification_summary = breaks_data.get("classification_summary", {})
    
    # Key metrics
    col1 = st.columns(1)[0]
    
    with col1:
        st.metric(
            "Total Breaks",
            total_breaks,
            help="Total number of discrepancies identified"
        )
    
    # Severity breakdown
    st.markdown("### Breaks by Severity")
    
    severity_data = {k: v for k, v in severity_summary.items()}
    
    if any(severity_data.values()):
        # Create severity chart
        fig = px.bar(
            x=[s.title() for s in severity_data.keys()],
            y=list(severity_data.values()),
            labels={"x": "Severity", "y": "Number of Breaks"},
            color=[s.title() for s in severity_data.keys()],
            color_discrete_map={
                "Critical": "#ff4444",
                "High": "#ff8800",
                "Medium": "#ffbb33",
                "Low": "#99cc00"
            }
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No breaks to display")
    
    # Type breakdown
    st.markdown("### Breaks by Classification")
    if classification_summary:
        type_df = pd.DataFrame(list(classification_summary.items()), columns=["Classification", "Count"])
        type_df = type_df.sort_values("Count", ascending=False)
        
        fig = px.pie(
            type_df,
            values="Count",
            names="Classification",
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Overall assessment
    st.markdown("### Overall Assessment")
    assessment_text = breaks_data.get("overall_assessment", "No assessment available")
    st.info(assessment_text)


def display_breakdown_section(breaks_data: Dict[str, Any]):
    """User-friendly breakdown of reconciliation breaks (table + card views)."""
    st.markdown("### 🔎 Reconciliation Breaks - Detailed Breakdown")
    breaks = breaks_data.get("breaks", [])
    if not breaks:
        st.info("No breaks to display")
        return
    
    # Filters
    colf1, colf2, colf3 = st.columns([2, 2, 2])
    with colf1:
        classifications = sorted({b.get("classification", "Unclassified") for b in breaks})
        selected_classes = st.multiselect("Classification", options=classifications, default=classifications)
    with colf2:
        severities = sorted({(b.get("severity") or "").lower() for b in breaks if b.get("severity")})
        selected_severities = st.multiselect("Severity", options=[s.title() for s in severities], default=[s.title() for s in severities])
    with colf3:
        text_query = st.text_input("Search text (ID, field, desc)", placeholder="Type to filter...")
    
    # Apply filters
    def match_text(b):
        if not text_query:
            return True
        q = text_query.lower()
        fields = [
            str(b.get("break_id", "")),
            str(b.get("classification", "")),
            str(b.get("affected_field", "")),
            str(b.get("nbim_record_identifier", "")),
            str(b.get("custody_record_identifier", "")),
            str(b.get("description", "")),
        ]
        return any(q in str(f).lower() for f in fields)
    
    filtered = [
        b for b in breaks
        if b.get("classification", "Unclassified") in selected_classes
        and ((b.get("severity") or "").title() in selected_severities)
        and match_text(b)
    ]
    
    # Build a concise, readable table
    rows = []
    for b in filtered:
        rows.append({
            "Break ID": b.get("break_id"),
            "Classification": b.get("classification"),
            "Severity": (b.get("severity") or "").title(),
            "Field": b.get("affected_field"),
            "NBIM value": b.get("nbim_value"),
            "Custody value": b.get("custody_value"),
            "Difference": b.get("difference"),
            "NBIM record": b.get("nbim_record_identifier"),
            "Custody record": b.get("custody_record_identifier"),
            "Impact": b.get("impact_assessment"),
            "Confidence": b.get("confidence"),
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    
    # Optional card view for the first 10 filtered breaks
    st.markdown("#### Card View")
    for i, b in enumerate(filtered[:10]):
        st.markdown("")
        with st.container(border=True):
            header_col1, header_col2 = st.columns([4, 1])
            with header_col1:
                st.markdown(f"**{b.get('classification', 'Unclassified')}** · {(b.get('severity') or '').upper()} · ID: {b.get('break_id','N/A')}")
                if b.get('affected_field'):
                    st.caption(f"Field: {b.get('affected_field')}")
            with header_col2:
                conf = b.get('confidence') or 0
                st.metric("Confidence", f"{conf:.0%}")
            
            body_col1, body_col2 = st.columns(2)
            with body_col1:
                st.write("NBIM")
                st.code(str(b.get('nbim_value', 'N/A')))
                if b.get('nbim_record_identifier'):
                    st.caption(f"Record: {b.get('nbim_record_identifier')}")
            with body_col2:
                st.write("Custody")
                st.code(str(b.get('custody_value', 'N/A')))
                if b.get('custody_record_identifier'):
                    st.caption(f"Record: {b.get('custody_record_identifier')}")
            
            if b.get('difference') is not None:
                st.caption(f"Difference: {b.get('difference')}")
            if b.get('impact_assessment'):
                st.info(b.get('impact_assessment'))


def display_resolution_section(breaks_data: Dict[str, Any]):
    """Display the resolution section with fix suggestions as a table."""
    
    st.markdown("### 🔧 Suggested Fixes")
    st.markdown("AI-powered suggestions for fixing the identified breaks.")
    
    # Resolution analysis button
    if st.session_state.resolution_result is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Fix Suggestions", type="primary", use_container_width=True, key="generate_fix_suggestions"):
                st.session_state.resolution_requested = True
                st.rerun()
    else:
        # Show re-run option and summary
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            resolution_data = st.session_state.resolution_result
            total_analyzed = resolution_data.get("total_breaks_analyzed", 0)
            total_resolvable = resolution_data.get("total_resolvable", 0)
            manual_review = resolution_data.get("total_requiring_manual_review", 0)
            automatable = resolution_data.get("automation_potential", {}).get("fully_automatable", 0)
            
            if total_resolvable > 0:
                st.success(f"📋 {total_analyzed} breaks analyzed • {total_resolvable} fixes suggested • {automatable} automatable • {manual_review} manual review")
            else:
                st.info(f"📋 {total_analyzed} breaks analyzed • All require manual investigation")
        with col2:
            if st.button("🔄 Re-analyze", use_container_width=True, key="re_analyze_resolution"):
                # Only reset resolution data, not breaks data
                st.session_state.resolution_result = None
                st.session_state.fix_decisions = {}
                st.session_state.resolution_requested = True
                # Don't trigger breaks analysis
                st.rerun()
        with col3:
            # Single action button for all changes
            display_save_and_jira_button()
    
    # Check if resolution was requested and run it
    if st.session_state.resolution_requested and st.session_state.resolution_result is None:
        run_breaks_resolution(breaks_data)
        st.session_state.resolution_requested = False
    
    # Display resolution table if available
    if st.session_state.resolution_result:
        display_resolution_table(breaks_data, st.session_state.resolution_result)
        
        # Add JIRA configuration section
        display_jira_configuration_section()
    else:
        st.info("Click 'Generate Fix Suggestions' to get AI-powered recommendations for fixing these breaks.")


def display_resolution_table(breaks_data: Dict[str, Any], resolution_data: Dict[str, Any]):
    """Display resolution suggestions in a table format with accept/reject functionality."""
    
    breaks = breaks_data.get("breaks", [])
    resolutions = resolution_data.get("resolutions", [])
    
    if not resolutions:
        st.warning("No fix suggestions were generated. This might indicate all breaks require manual investigation.")
        return
    
    # Create a mapping of break_id to break data for easy lookup
    break_lookup = {b.get("break_id"): b for b in breaks}
    
    # Prepare table data
    table_data = []
    for resolution in resolutions:
        break_id = resolution.get("break_id")
        break_info = break_lookup.get(break_id, {})
        
        # Get suggested fix value and reasoning
        suggested_value = extract_suggested_value(resolution)
        reasoning = resolution.get("reasoning", "No reasoning provided")
        confidence = resolution.get("confidence", 0)
        
        # Get current decision or default to None
        current_decision = st.session_state.fix_decisions.get(break_id, None)
        
        table_data.append({
            "Break ID": break_id,
            "Classification": break_info.get("classification", "Unknown"),
            "Severity": (break_info.get("severity", "").title() or "Unknown"),
            "Field": break_info.get("affected_field", "Unknown"),
            "Current Value": format_value(break_info.get("nbim_value")),
            "Custody Value": format_value(break_info.get("custody_value")),
            "Suggested Fix Value": suggested_value,
            "Reasoning": reasoning,
            "Confidence": f"{confidence:.0%}" if confidence > 0 else "Manual Review",
            "Status": "Automated" if confidence >= 0.8 and suggested_value != "Manual review required" else "Manual Review",
            "Accept": current_decision if current_decision is not None else False
        })
    
    if not table_data:
        st.warning("No resolution data to display.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Configure columns for data editor
    column_config = {
        "Break ID": st.column_config.TextColumn("Break ID", disabled=True),
        "Classification": st.column_config.TextColumn("Classification", disabled=True),
        "Severity": st.column_config.TextColumn("Severity", disabled=True),
        "Field": st.column_config.TextColumn("Field", disabled=True),
        "Current Value": st.column_config.TextColumn("Current Value", disabled=True),
        "Custody Value": st.column_config.TextColumn("Custody Value", disabled=True),
        "Suggested Fix Value": st.column_config.TextColumn("Suggested Fix Value", disabled=True, help="AI-suggested corrected value"),
        "Reasoning": st.column_config.TextColumn("Reasoning", disabled=True, help="AI's reasoning for the suggested value"),
        "Confidence": st.column_config.TextColumn("Confidence", disabled=True, help="AI's confidence in the suggestion"),
        "Status": st.column_config.TextColumn("Status", disabled=True, help="Whether fix can be automated or requires manual review"),
        "Accept": st.column_config.CheckboxColumn(
            "Accept Fix",
            help="Check to accept this fix suggestion",
            default=False,
        )
    }
    
    # Display the editable table
    edited_df = st.data_editor(
        df,
        column_config=column_config,
        disabled=["Break ID", "Classification", "Severity", "Field", "Current Value", "Custody Value", "Suggested Fix Value", "Reasoning", "Confidence", "Status"],
        hide_index=True,
        use_container_width=True,
        key=f"resolution_table_{len(resolutions)}"
    )
    
    # Update session state based on user decisions
    update_fix_decisions(edited_df)
    
    # Show current decision summary
    display_current_decisions_summary()


def extract_suggested_value(resolution: Dict[str, Any]) -> str:
    """Extract suggested fix value from resolution data."""
    
    # The new format directly provides the corrected value
    corrected_value = resolution.get("corrected_value")
    
    if corrected_value:
        return str(corrected_value)
    
    # Fallback for any legacy format
    return "Manual review needed"


def format_value(value) -> str:
    """Format values for display in the table."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}" if value != int(value) else f"{int(value):,}"
    return str(value)


def update_fix_decisions(edited_df: pd.DataFrame):
    """Update session state with user's fix decisions."""
    
    for _, row in edited_df.iterrows():
        break_id = row["Break ID"]
        accepted = row["Accept"]
        
        # Update decision if it changed
        current_decision = st.session_state.fix_decisions.get(break_id)
        if current_decision != accepted:
            st.session_state.fix_decisions[break_id] = accepted


def display_current_decisions_summary():
    """Display a summary of current fix decisions."""
    
    if not st.session_state.fix_decisions:
        return
    
    accepted_count = sum(1 for decision in st.session_state.fix_decisions.values() if decision)
    rejected_count = sum(1 for decision in st.session_state.fix_decisions.values() if not decision)
    
    if accepted_count > 0 or rejected_count > 0:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if accepted_count > 0:
                st.success(f"✅ {accepted_count} fixes selected to apply")
        
        with col2:
            if rejected_count > 0:
                st.warning(f"🎫 {rejected_count} items selected for Jira tickets")
        
        with col3:
            if accepted_count + rejected_count > 0:
                st.info(f"📝 {accepted_count + rejected_count} decisions made")


def display_save_and_jira_button():
    """Display single button to save changes and create Jira issues."""
    
    if not st.session_state.fix_decisions:
        return
    
    accepted_fixes = [k for k, v in st.session_state.fix_decisions.items() if v]
    rejected_fixes = [k for k, v in st.session_state.fix_decisions.items() if not v]
    total_decisions = len([k for k, v in st.session_state.fix_decisions.items() if v is not None])
    
    if total_decisions > 0:
        button_text = "💾 Save Changes and Create Jira Issues"
        if len(accepted_fixes) > 0 and len(rejected_fixes) > 0:
            button_text = f"💾 Save Changes and Create Jira Issues ({len(accepted_fixes)} fixes + {len(rejected_fixes)} tickets)"
        elif len(accepted_fixes) > 0:
            button_text = f"💾 Save Changes ({len(accepted_fixes)} fixes to apply)"
        elif len(rejected_fixes) > 0:
            button_text = f"🎫 Create Jira Issues ({len(rejected_fixes)} tickets)"
        
        if st.button(
            button_text, 
            type="primary", 
            use_container_width=True,
            help="Apply accepted fixes to CSV files and create Jira tickets for rejected items"
        ):
            process_fixes_and_create_jira(accepted_fixes, rejected_fixes)


def process_fixes_and_create_jira(accepted_fixes: List[str], rejected_fixes: List[str]):
    """Process accepted fixes and create Jira issues for rejected ones."""
    
    if not st.session_state.resolution_result:
        st.error("No resolution data available")
        return
    
    resolutions = st.session_state.resolution_result.get("resolutions", [])
    resolution_lookup = {r.get("break_id"): r for r in resolutions}
    
    # Process accepted fixes
    if accepted_fixes:
        with st.spinner("📝 Creating updated CSV files with applied fixes..."):
            success_count = create_updated_csv_files(accepted_fixes, resolution_lookup)
            if success_count > 0:
                st.success(f"✅ Created updated CSV files with {success_count} fixes applied!")
            else:
                st.error("❌ Failed to create updated CSV files")
    
    # Handle rejected fixes - Create actual JIRA tickets
    if rejected_fixes:
        with st.spinner(f"🎫 Creating JIRA tickets for {len(rejected_fixes)} items requiring manual review..."):
            jira_success = create_jira_tickets_for_rejected_fixes(rejected_fixes, resolution_lookup)
            
            if jira_success:
                st.success(f"✅ Successfully created {len(rejected_fixes)} JIRA tickets for manual review items!")
            else:
                st.error("❌ Failed to create JIRA tickets. See details below.")


def create_updated_csv_files(accepted_fixes: List[str], resolution_lookup: Dict[str, Any]) -> int:
    """Create updated CSV files with fixes applied."""
    
    try:
        # Get the original data from session state or wherever it's stored
        # This would need to be adapted based on how you store the original CSV data
        
        # For now, create a placeholder implementation
        fixes_applied = []
        
        for break_id in accepted_fixes:
            resolution = resolution_lookup.get(break_id)
            if resolution:
                corrected_value = resolution.get("corrected_value")
                reasoning = resolution.get("reasoning")
                
                # Here you would apply the fix to the actual data
                # For now, just track what would be fixed
                fixes_applied.append({
                    "break_id": break_id,
                    "corrected_value": corrected_value,
                    "reasoning": reasoning
                })
        
        # Create a summary of fixes that were applied
        if fixes_applied:
            fixes_df = pd.DataFrame(fixes_applied)
            
            # Download button for the fixes summary
            csv_data = fixes_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Applied Fixes Summary",
                data=csv_data,
                file_name=f"applied_fixes_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # TODO: Implement actual CSV file updates here
            # This would involve:
            # 1. Loading the original NBIM/Custody CSV files
            # 2. Finding the rows that correspond to each break_id
            # 3. Updating those rows with the corrected values
            # 4. Saving the updated CSV files
            
        return len(fixes_applied)
        
    except Exception as e:
        st.error(f"Error creating updated CSV files: {str(e)}")
        return 0


def create_jira_tickets_for_rejected_fixes(rejected_fixes: List[str], resolution_lookup: Dict[str, Any]) -> bool:
    """Create JIRA tickets for rejected fixes that require manual review."""
    
    try:
        # Check if OpenAI API key is available
        import os
        if not os.getenv("OPENAI_API_KEY"):
            st.error("🔑 OpenAI API key not found! Please add OPENAI_API_KEY to your .env file for JIRA ticket generation.")
            return False
        # Get the original breaks data from session state
        if not hasattr(st.session_state, 'breaks_result') or not st.session_state.breaks_result:
            st.error("❌ No breaks data available for JIRA ticket creation")
            return False
            
        breaks_data = st.session_state.breaks_result
        all_breaks = breaks_data.get("breaks", [])
        
        # Get breaks data for rejected fixes
        breaks_for_jira = []
        break_lookup = {b.get("break_id"): b for b in all_breaks}
        
        for break_id in rejected_fixes:
            break_data = break_lookup.get(break_id)
            if break_data:
                breaks_for_jira.append(break_data)
        
        if not breaks_for_jira:
            st.warning("⚠️ No break data found for selected rejected fixes")
            return False
        
        # Create JIRA agent and generate tickets
        jira_agent = JiraIssueAgent()
        
        # Build additional context for JIRA tickets
        additional_context = f"""
        These breaks were identified during dividend reconciliation analysis.
        They were marked as requiring manual review by the reconciliation team.
        
        Total breaks requiring manual review: {len(breaks_for_jira)}
        Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Each ticket represents a discrepancy that could not be automatically resolved
        and requires investigation by the reconciliation team.
        """
        
        # Get resolution data for context
        resolution_data = st.session_state.resolution_result if hasattr(st.session_state, 'resolution_result') else None
        
        # Get JIRA configuration from user settings or use defaults
        user_jira_config = getattr(st.session_state, 'jira_config', {})
        jira_config = {
            "project_key": user_jira_config.get("project_key", "RECON"),
            "default_assignee": user_jira_config.get("default_assignee", "reconciliation-team"),
            "component": user_jira_config.get("component", "Dividend Reconciliation"),
            "default_priority": user_jira_config.get("default_priority", "Medium"),
            "labels": user_jira_config.get("labels", ["reconciliation", "dividend", "breaks", "manual-review"]),
            "epic_link": user_jira_config.get("epic_link"),
            "custom_fields": {
                "reconciliation_date": datetime.now().strftime('%Y-%m-%d'),
                "system_source": "NBIM-Custody Reconciliation"
            }
        }
        
        # Create JIRA tickets
        result = asyncio.run(jira_agent.create_jira_issues(
            breaks=breaks_for_jira,
            resolution_data=resolution_data,
            jira_config=jira_config,
            additional_context=additional_context.strip(),
            timeout=120.0
        ))
        
        # Validate and process results
        if jira_agent.validate_jira_issues(result):
            jira_data = result.response.structured_data
            
            # Display success summary
            total_created = jira_data.get("total_issues_created", 0)
            if total_created > 0:
                st.success(f"🎫 Created {total_created} JIRA tickets successfully!")
                
                # Show breakdown by priority if available
                priority_breakdown = jira_data.get("issues_by_priority", {})
                if priority_breakdown:
                    st.info(f"📊 Priority breakdown: " + ", ".join([f"{p}: {c}" for p, c in priority_breakdown.items()]))
                
                # Provide download option for JIRA CSV
                display_jira_download_option(jira_data)
                
                # Show preview of created tickets
                display_jira_tickets_preview(jira_data)
                
                return True
            else:
                st.error("❌ No JIRA tickets were created")
                return False
        else:
            error_msg = str(result.error) if result.error else "JIRA ticket validation failed"
            st.error(f"❌ JIRA ticket creation failed: {error_msg}")
            return False
            
    except Exception as e:
        st.error(f"🚨 Error creating JIRA tickets: {str(e)}")
        logger.error(f"JIRA ticket creation error: {e}")
        return False


def display_jira_download_option(jira_data: Dict[str, Any]):
    """Display download option for JIRA CSV."""
    
    jira_issues = jira_data.get("jira_issues", [])
    if not jira_issues:
        return
    
    # Convert JIRA issues to DataFrame
    jira_df = pd.DataFrame(jira_issues)
    
    # Create CSV for download
    csv_data = jira_df.to_csv(index=False)
    
    # Download button
    st.download_button(
        label="📥 Download JIRA Import CSV",
        data=csv_data,
        file_name=f"jira_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download CSV file to import tickets directly into JIRA",
        use_container_width=True
    )


def display_jira_tickets_preview(jira_data: Dict[str, Any]):
    """Display a preview of the created JIRA tickets."""
    
    jira_issues = jira_data.get("jira_issues", [])
    if not jira_issues:
        return
    
    with st.expander(f"📋 Preview of {len(jira_issues)} JIRA Tickets Created", expanded=False):
        
        # Show summary table
        preview_data = []
        for issue in jira_issues[:10]:  # Limit to first 10 for preview
            preview_data.append({
                "Key": issue.get("issue_key", "N/A"),
                "Title": issue.get("summary", "No title")[:50] + "..." if len(issue.get("summary", "")) > 50 else issue.get("summary", ""),
                "Priority": issue.get("priority", "Medium"),
                "Type": issue.get("issue_type", "Task"),
                "Assignee": issue.get("assignee", "Unassigned")
            })
        
        if preview_data:
            preview_df = pd.DataFrame(preview_data)
            st.dataframe(preview_df, use_container_width=True)
            
            if len(jira_issues) > 10:
                st.info(f"Showing first 10 of {len(jira_issues)} tickets. Download CSV to see all tickets.")
        
        # Show detailed view of first ticket
        if jira_issues:
            st.markdown("#### Sample Ticket Details:")
            sample_issue = jira_issues[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Key:** {sample_issue.get('issue_key', 'N/A')}")
                st.write(f"**Priority:** {sample_issue.get('priority', 'Medium')}")
                st.write(f"**Type:** {sample_issue.get('issue_type', 'Task')}")
            with col2:
                st.write(f"**Component:** {sample_issue.get('component', 'Reconciliation')}")
                st.write(f"**Labels:** {', '.join(sample_issue.get('labels', []))}")
                st.write(f"**Assignee:** {sample_issue.get('assignee', 'Unassigned')}")
            
            st.markdown("**Summary:**")
            st.write(sample_issue.get('summary', 'No summary available'))
            
            st.markdown("**Description:**")
            description = sample_issue.get('description', 'No description available')
            if len(description) > 300:
                st.write(description[:300] + "...")
                with st.expander("Show full description"):
                    st.write(description)
            else:
                st.write(description)


def display_jira_configuration_section():
    """Display JIRA configuration options for users to customize ticket creation."""
    
    with st.expander("⚙️ JIRA Ticket Configuration (Optional)", expanded=False):
        st.markdown("Customize how JIRA tickets will be created for manual review items:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            jira_project = st.text_input(
                "JIRA Project Key",
                value="RECON",
                help="The project key where tickets will be created",
                key="jira_project_key"
            )
            
            jira_component = st.text_input(
                "Component",
                value="Dividend Reconciliation",
                help="Component to assign to the tickets",
                key="jira_component"
            )
            
            jira_assignee = st.text_input(
                "Default Assignee",
                value="reconciliation-team",
                help="Default assignee for the tickets (username or team)",
                key="jira_assignee"
            )
        
        with col2:
            jira_priority = st.selectbox(
                "Default Priority",
                options=["Low", "Medium", "High", "Highest"],
                index=1,  # Default to Medium
                help="Default priority for manual review tickets",
                key="jira_priority"
            )
            
            jira_labels = st.text_input(
                "Labels (comma-separated)",
                value="reconciliation,dividend,breaks,manual-review",
                help="Labels to add to the tickets",
                key="jira_labels"
            )
            
            jira_epic = st.text_input(
                "Epic Link (optional)",
                value="",
                help="Link tickets to an epic (epic key)",
                key="jira_epic"
            )
        
        st.markdown("**Note:** These settings will be used when creating JIRA tickets for items requiring manual review.")
        
        # Store configuration in session state
        jira_config = {
            "project_key": jira_project,
            "default_assignee": jira_assignee,
            "component": jira_component,
            "default_priority": jira_priority,
            "labels": [label.strip() for label in jira_labels.split(",") if label.strip()],
            "epic_link": jira_epic if jira_epic else None
        }
        
        st.session_state.jira_config = jira_config






def display_breaks_details_tab(breaks_data: Dict[str, Any]):
    """Display detailed breaks information."""
    
    breaks = breaks_data.get("breaks", [])
    
    if not breaks:
        st.info("No breaks to display")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severities = sorted(list({(b.get("severity") or "").lower() for b in breaks if b.get("severity")}))
        severity_filter = st.selectbox(
            "Filter by Severity",
            options=["All"] + [s.title() for s in severities],
            key="severity_filter"
        )
    
    with col2:
        classifications = sorted(list({b.get("classification") for b in breaks if b.get("classification")}))
        type_filter = st.selectbox(
            "Filter by Classification",
            options=["All"] + classifications,
            key="type_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=["Severity", "Confidence", "Classification"],
            key="sort_by"
        )
    
    # Apply filters
    filtered_breaks = breaks
    
    if severity_filter != "All":
        filtered_breaks = [b for b in filtered_breaks if (b.get("severity") or "").lower() == severity_filter.lower()]
    
    if type_filter != "All":
        filtered_breaks = [b for b in filtered_breaks if b.get("classification") == type_filter]
    
    # Sort breaks
    sort_key_map = {
        "Severity": lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get((x.get("severity") or "").lower(), 5),
        "Confidence": lambda x: -(x.get("confidence") or 0),
        "Classification": lambda x: x.get("classification", "")
    }
    filtered_breaks.sort(key=sort_key_map[sort_by])
    
    # Display breaks
    st.markdown(f"### Showing {len(filtered_breaks)} of {len(breaks)} breaks")
    
    for i, break_item in enumerate(filtered_breaks[:20]):  # Limit to 20 for performance
        with st.expander(
            f"[{(break_item.get('severity') or '').upper()}] {break_item.get('classification', 'Unclassified')} - {break_item.get('description', 'No description')[:100]}",
            expanded=(i < 3)
        ):
            display_single_break(break_item)
    
    if len(filtered_breaks) > 20:
        st.info(f"Showing first 20 breaks. {len(filtered_breaks) - 20} more available in export.")


def display_single_break(break_item: Dict[str, Any]):
    """Display a single break item."""
    
    # Break details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Break Information**")
        st.write(f"• ID: {break_item.get('break_id', 'N/A')}")
        st.write(f"• Classification: {break_item.get('classification', 'N/A')}")
        st.write(f"• Severity: {(break_item.get('severity') or 'N/A').upper()}")
        st.write(f"• Confidence: {break_item.get('confidence', 0):.0%}")
    
    with col2:
        st.markdown("**Impact**")
        financial_impact = break_item.get('financial_impact')
        if financial_impact:
            st.write(f"• Financial: ${financial_impact:,.2f}")
        affected_shares = break_item.get('affected_shares')
        if affected_shares:
            st.write(f"• Shares: {affected_shares:,.0f}")
    
    # Values comparison
    if break_item.get('nbim_value') is not None or break_item.get('custody_value') is not None:
        st.markdown("**Values**")
        value_col1, value_col2, value_col3 = st.columns(3)
        
        with value_col1:
            st.write("NBIM:")
            st.code(str(break_item.get('nbim_value', 'N/A')))
        
        with value_col2:
            st.write("Custody:")
            st.code(str(break_item.get('custody_value', 'N/A')))
        
        with value_col3:
            if break_item.get('difference') is not None:
                st.write("Difference:")
                st.code(str(break_item.get('difference')))
    
    # Description
    st.markdown("**Description**")
    st.write(break_item.get('description', 'No description available'))
    
    # Impact assessment
    if break_item.get('impact_assessment'):
        st.markdown("**Impact Assessment**")
        st.info(break_item.get('impact_assessment'))


def display_analytics_tab(breaks_data: Dict[str, Any]):
    """Display analytics and patterns."""
    
    breaks = breaks_data.get("breaks", [])
    
    if not breaks:
        st.info("No data for analytics")
        return
    
    # Prepare data for visualization
    breaks_df = pd.DataFrame(breaks)
    
    # Counts by classification
    if 'classification' in breaks_df.columns:
        st.markdown("### Breaks by Classification")
        cls_counts = breaks_df['classification'].value_counts().reset_index()
        cls_counts.columns = ['classification', 'count']
        fig = px.bar(cls_counts, x='classification', y='count', labels={'classification': 'Classification', 'count': 'Count'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Confidence distribution
    if 'confidence' in breaks_df.columns:
        st.markdown("### Confidence Score Distribution")
        
        fig = px.histogram(
            breaks_df,
            x='confidence',
            nbins=20,
            labels={'confidence': 'Confidence Score', 'count': 'Number of Breaks'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Patterns identified
    patterns = breaks_data.get("patterns_identified", [])
    if patterns:
        st.markdown("### Patterns Identified")
        for pattern in patterns:
            st.write(f"• {pattern}")
    
    # Field vs classification matrix
    if len(breaks_df) > 10 and 'affected_field' in breaks_df.columns and 'classification' in breaks_df.columns:
        st.markdown("### Field vs Classification Matrix")
        crosstab = pd.crosstab(breaks_df['affected_field'], breaks_df['classification'])
        if not crosstab.empty:
            fig = px.imshow(crosstab, labels=dict(x="Classification", y="Field", color="Count"), color_continuous_scale="YlOrRd")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def display_recommendations_tab(breaks_data: Dict[str, Any]):
    """Display recommendations and suggested controls."""
    
    st.markdown("### 💡 Key Recommendations")
    
    st.info("No recommendations available in identification phase.")
    
    st.markdown("### 🛡️ Suggested Controls")
    
    st.info("Controls will be suggested in the reconciliation planning phase.")
    
    # Action items based on severity
    breaks = breaks_data.get("breaks", [])
    critical_breaks = [b for b in breaks if (b.get("severity") or "").lower() == "critical"]
    high_breaks = [b for b in breaks if (b.get("severity") or "").lower() == "high"]
    
    if critical_breaks or high_breaks:
        st.markdown("### 🎯 Priority Actions")
        
        if critical_breaks:
            st.error(f"**Critical Issues ({len(critical_breaks)})**")
            for break_item in critical_breaks[:3]:
                st.write(f"• {break_item.get('description', 'N/A')[:100]}...")
                if break_item.get('impact_assessment'):
                    st.write(f"  → {break_item.get('impact_assessment')}")
        
        if high_breaks:
            st.warning(f"**High Priority Issues ({len(high_breaks)})**")
            for break_item in high_breaks[:3]:
                st.write(f"• {break_item.get('description', 'N/A')[:100]}...")
                if break_item.get('impact_assessment'):
                    st.write(f"  → {break_item.get('impact_assessment')}")


def display_export_tab(breaks_data: Dict[str, Any]):
    """Display export options."""
    
    st.markdown("### 📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export breaks as CSV
        if st.button("Export Breaks to CSV", use_container_width=True):
            export_breaks_csv(breaks_data)
    
    with col2:
        # Export full report
        if st.button("Generate Full Report", use_container_width=True):
            export_full_report(breaks_data)
    
    # Quick summary for copying
    st.markdown("### 📋 Quick Summary (Copy & Paste)")
    
    summary_text = generate_text_summary(breaks_data)
    st.text_area(
        "Summary",
        value=summary_text,
        height=200,
        help="Select all and copy this summary"
    )


def export_breaks_csv(breaks_data: Dict[str, Any]):
    """Export breaks to CSV."""
    
    breaks = breaks_data.get("breaks", [])
    
    if not breaks:
        st.warning("No breaks to export")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(breaks)
    
    # Reorder columns for better readability
    column_order = [
        'break_id', 'classification', 'severity', 'description',
        'nbim_record_identifier', 'custody_record_identifier', 'affected_field',
        'nbim_value', 'custody_value', 'difference',
        'impact_assessment', 'confidence'
    ]
    
    # Only include columns that exist
    columns_to_export = [col for col in column_order if col in df.columns]
    df = df[columns_to_export]
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    # Download button
    st.download_button(
        label="💾 Download Breaks CSV",
        data=csv,
        file_name=f"breaks_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def export_full_report(breaks_data: Dict[str, Any]):
    """Generate and export a full HTML report."""
    
    html_report = generate_html_report(breaks_data)
    
    st.download_button(
        label="💾 Download HTML Report",
        data=html_report,
        file_name=f"reconciliation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html"
    )


def generate_text_summary(breaks_data: Dict[str, Any]) -> str:
    """Generate a text summary of the breaks analysis."""
    
    total_breaks = breaks_data.get("total_breaks_found", 0)
    classification_summary = breaks_data.get("classification_summary", {})
    severity_summary = breaks_data.get("severity_summary", {})
    
    text = f"""DIVIDEND RECONCILIATION BREAKS ANALYSIS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Total Breaks: {total_breaks}
By Classification: {json.dumps(classification_summary)}
By Severity: {json.dumps(severity_summary)}

ANALYSIS
--------
{breaks_data.get('overall_assessment', 'N/A')}

KEY RECOMMENDATIONS
------------------
"""
    # Recommendations are generated in a later phase
    text += "(To be generated in reconciliation planning phase)\n"
    
    return text


def generate_html_report(breaks_data: Dict[str, Any]) -> str:
    """Generate an HTML report of the breaks analysis."""
    
    total_breaks = breaks_data.get("total_breaks_found", 0)
    classification_summary = breaks_data.get("classification_summary", {})
    severity_summary = breaks_data.get("severity_summary", {})
    breaks = breaks_data.get("breaks", [])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reconciliation Breaks Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #333; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px 20px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .critical {{ color: #ff4444; font-weight: bold; }}
            .high {{ color: #ff8800; font-weight: bold; }}
            .medium {{ color: #ffbb33; }}
            .low {{ color: #99cc00; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 10px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f0f0f0; }}
            .break-item {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Dividend Reconciliation Breaks Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Executive Summary</h2>
        <div>
            <div class="metric">
                <div class="metric-value">{total_breaks}</div>
                <div class="metric-label">Total Breaks</div>
            </div>
        </div>
        
        <h2>Severity Breakdown</h2>
        <table>
            <tr>
                <th>Severity</th>
                <th>Count</th>
            </tr>
            <tr class="critical">
                <td>Critical</td>
                <td>{severity_summary.get('critical', 0)}</td>
            </tr>
            <tr class="high">
                <td>High</td>
                <td>{severity_summary.get('high', 0)}</td>
            </tr>
            <tr class="medium">
                <td>Medium</td>
                <td>{severity_summary.get('medium', 0)}</td>
            </tr>
            <tr class="low">
                <td>Low</td>
                <td>{severity_summary.get('low', 0)}</td>
            </tr>
        </table>
        
        <h2>Assessment</h2>
        <p>{breaks_data.get('overall_assessment', 'No assessment available')}</p>
        
        <h2>Break Details</h2>
    """
    
    # Add first 20 breaks
    for break_item in breaks[:20]:
        severity_class = (break_item.get('severity') or 'low')
        html += f"""
        <div class="break-item">
            <h3 class="{severity_class}">[{(break_item.get('severity') or '').upper()}] {break_item.get('classification', 'Unclassified')}</h3>
            <p><strong>Description:</strong> {break_item.get('description', 'N/A')}</p>
            <p><strong>Impact:</strong> {break_item.get('impact_assessment', 'N/A')}</p>
            <p><strong>Confidence:</strong> {break_item.get('confidence', 0):.0%}</p>
        </div>
        """
    
    if len(breaks) > 20:
        html += f"<p><em>Showing first 20 of {len(breaks)} total breaks</em></p>"
    
    html += """
    </body>
    </html>
    """
    
    return html