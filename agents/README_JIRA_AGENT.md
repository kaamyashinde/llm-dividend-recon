# JIRA Issue Agent

The JIRA Issue Agent converts reconciliation breaks into JIRA-compatible tickets that can be imported and tracked by your reconciliation team.

## Overview

When reconciliation breaks require manual investigation, the JIRA agent creates properly formatted JIRA issues with:
- Clear, actionable summaries
- Detailed descriptions with break context
- Appropriate priority and issue type mapping
- Custom fields for reconciliation tracking
- Export options for CSV import or API integration

## Features

### ✅ Smart Mapping
- **Severity → Priority**: Critical→Highest, High→High, Medium→Medium, Low→Low
- **Classification → Issue Type**: Value Discrepancy→Task, Missing Record→Bug, etc.
- **Context Preservation**: All break details included in JIRA description

### ✅ Multiple Export Formats
- **CSV Import**: Ready for JIRA's bulk import feature
- **JSON API**: Formatted for JIRA REST API integration
- **Custom Fields**: Break ID, NBIM/Custody values, financial impact

### ✅ Streamlit Integration
- Seamless integration with the reconciliation workflow
- Visual preview of JIRA issues before creation
- Download buttons for immediate export

## Usage

### In Streamlit App

1. **Run Reconciliation**: Upload files and identify breaks
2. **Generate Fixes**: Use the resolution agent to get fix suggestions
3. **Make Decisions**: Accept fixes that can be automated, reject those needing investigation
4. **Create JIRA Issues**: Rejected items automatically become JIRA tickets

```python
# Items marked as "rejected" in the resolution table will trigger JIRA creation
# Click "Save Changes and Create JIRA Issues" button
```

### Direct API Usage

```python
from agents.jira_issue_agent import JiraIssueAgent

# Sample break data
breaks = [{
    "break_id": "BRK_001",
    "description": "Tax amount mismatch between systems",
    "classification": "Value Discrepancy",
    "severity": "High",
    "affected_field": "withholding_tax",
    "nbim_value": 360.00,
    "custody_value": 300.00,
    "impact_assessment": "Net settlement incorrect by $60"
}]

# Create JIRA agent
agent = JiraIssueAgent()

# Generate JIRA issues
result = await agent.create_jira_issues(
    breaks=breaks,
    resolution_data=resolution_data,  # Optional
    jira_config={"project_key": "RECON"}  # Optional
)

# Export formats
csv_data = agent.format_for_csv_import(result)
json_data = agent.format_for_json_import(result)
```

## JIRA Configuration

### Required JIRA Setup

1. **Create Project**: Set up a JIRA project (e.g., "RECON") 
2. **Configure Issue Types**: Ensure Task, Bug, Story types exist
3. **Set Up Components**: Create "Dividend Reconciliation" component
4. **Custom Fields** (optional but recommended):
   - Break ID (Text)
   - Break Classification (Select)
   - NBIM Value (Text)
   - Custody Value (Text)
   - Financial Impact (Text)

### Import Process

#### CSV Import Method
1. Download CSV file from Streamlit app
2. Go to JIRA Project → Issues → Import issues from CSV
3. Map CSV columns to JIRA fields:
   - Summary → Summary
   - Description → Description
   - Issue Type → Issue Type
   - Priority → Priority
   - Break ID → Custom Field
4. Review mapping and import

#### API Method
1. Download JSON file from Streamlit app
2. Use JIRA REST API endpoint: `/rest/api/2/issue/bulk`
3. Ensure custom field IDs match your JIRA instance
4. Test with small batch first

## Sample Output

### JIRA Issue Example
```
Summary: Fix withholding tax calculation for Apple dividend
Type: Task
Priority: High
Component: Dividend Reconciliation
Labels: reconciliation, dividend, manual-review

Description:
## Break Details
- Break ID: BRK_001
- Classification: Value Discrepancy
- Field: withholding_tax
- NBIM Value: 360.00
- Custody Value: 300.00

## Impact Assessment
Net settlement amount will be incorrect by $60

## Required Action
Investigate tax rate applied to US securities. Standard rate should be 15%.
Review custodian confirmations and update system if needed.
```

### CSV Export Sample
```csv
Summary,Description,Issue Type,Priority,Component,Labels,Break ID,Break Classification
"Fix withholding tax calculation for Apple dividend","## Break Details...",Task,High,"Dividend Reconciliation","reconciliation;dividend;manual-review",BRK_001,"Value Discrepancy"
```

## Agent Configuration

### Default Settings
```python
{
    "default_component": "Dividend Reconciliation",
    "default_labels": ["reconciliation", "dividend", "automated"],
    "priority_mapping": {
        "Critical": "Highest",
        "High": "High", 
        "Medium": "Medium",
        "Low": "Low"
    },
    "issue_type_mapping": {
        "Value Discrepancy": "Task",
        "Missing Record": "Bug",
        "System Error": "Bug",
        "Process Issue": "Improvement"
    }
}
```

### Custom Configuration
```python
jira_config = {
    "project_key": "RECON",
    "default_component": "Dividend Reconciliation",
    "default_labels": ["Q1-2024", "automated", "reconciliation"],
    "due_date_offset_days": 5,  # Issues due in 5 days
    "default_assignee": "recon-team@company.com"
}

result = await agent.create_jira_issues(
    breaks=breaks,
    jira_config=jira_config
)
```

## Integration Points

### With Breaks Identifier Agent
- Receives break data with classification and severity
- Uses composite keys for unique identification
- Preserves all break context and impact assessments

### With Resolution Agent  
- Uses resolution suggestions to enhance JIRA descriptions
- Distinguishes between automatable fixes and manual review items
- Only creates JIRA issues for items requiring manual intervention

### With Streamlit App
- Integrated into the reconciliation workflow
- Visual preview before JIRA creation
- Download buttons for immediate export
- Progress indicators and error handling

## Error Handling

The agent includes comprehensive error handling:

- **Validation**: Ensures all required fields are present
- **Fallback**: Shows preview even if JIRA creation fails
- **Logging**: Detailed logs for troubleshooting
- **Recovery**: Graceful degradation if API calls fail

## Best Practices

### For Reconciliation Teams
1. **Review Before Import**: Always preview JIRA issues before importing
2. **Batch Processing**: Import issues in small batches for better control
3. **Field Mapping**: Set up custom fields in JIRA before first import
4. **Automation Rules**: Create JIRA automation for status updates

### For Development
1. **Custom Fields**: Map `customfield_*` IDs to your JIRA instance
2. **Testing**: Test with small datasets first
3. **Templates**: Customize description templates for your organization
4. **Monitoring**: Set up logging for JIRA creation attempts

## Troubleshooting

### Common Issues

**"No JIRA issues created"**
- Check that breaks have required fields (break_id, description, classification)
- Verify OpenAI API key is set
- Review agent logs for specific errors

**"CSV import fails in JIRA"** 
- Ensure column mapping matches your JIRA configuration
- Check that issue types and priorities exist in your JIRA project
- Verify required fields are not empty

**"Custom fields not populating"**
- Update custom field IDs in JSON format to match your JIRA instance
- Ensure custom fields are configured for the issue type
- Check field permissions

### Support
- Check logs in `logs/` directory
- Review `test_jira_agent_simple.py` for validation
- Run example usage in `jira_issue_agent.py`

## Future Enhancements

- [ ] Direct JIRA API integration (create issues automatically)
- [ ] JIRA webhook integration for status updates
- [ ] Template customization for different break types  
- [ ] Bulk status updates based on resolution progress
- [ ] Integration with JIRA automation rules
- [ ] Dashboard for JIRA issue tracking