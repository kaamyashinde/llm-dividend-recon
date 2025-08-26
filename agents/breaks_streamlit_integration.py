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

# Import the breaks identifier agent
from agents.breaks_identifier_agent import BreaksIdentifierAgent
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
            help="Focus analysis on specific dividend event"
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
            help="Specific areas to focus the analysis on"
        )
    
    with col3:
        # Analysis depth
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Comprehensive"],
            value="Standard",
            help="Level of detail in the analysis"
        )
    
    # Run analysis button
    if st.session_state.breaks_result is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Identify Breaks", type="primary", use_container_width=True):
                run_breaks_analysis(nbim_df, custody_df, mappings, selected_event, focus_areas, analysis_depth)
    else:
        # Show re-run option
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🔄 Re-run Analysis", use_container_width=True):
                st.session_state.breaks_result = None
                st.rerun()
    
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


def display_breaks_results(breaks_data: Dict[str, Any]):
    """Display the breaks analysis results with a summary and a clear breakdown."""
    
    # Keep the summary as a dedicated tab
    tab1, = st.tabs(["📊 Summary"])
    with tab1:
        display_summary_tab(breaks_data)
    
    # Show a clearer breakdown below
    st.markdown("---")
    display_breakdown_section(breaks_data)


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