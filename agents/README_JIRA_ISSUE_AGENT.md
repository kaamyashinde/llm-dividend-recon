# 🎫 JIRA Issue Agent

> **Automated JIRA ticket creation for reconciliation breaks requiring manual investigation**

## 📋 Overview

The JIRA Issue Agent transforms reconciliation breaks into properly formatted JIRA tickets that can be directly imported into JIRA or created via API. It generates rich ticket descriptions with business context, impact analysis, and actionable investigation steps, ensuring reconciliation teams have all necessary information for efficient resolution.

## ✨ Key Features

### 🎯 **Intelligent Ticket Generation**
- **Rich Descriptions**: AI-generated ticket content with full business context
- **Priority Mapping**: Automatic priority assignment based on break severity
- **Impact Analysis**: Business impact assessment for each ticket
- **Investigation Steps**: Clear action items for reconciliation teams

### 📊 **Professional Formatting**
- **JIRA-Compatible**: Direct CSV import or API integration ready
- **Structured Content**: Consistent ticket format with sections and formatting
- **Metadata Inclusion**: All relevant reconciliation context and identifiers
- **Audit Trail**: Complete traceability back to original breaks

### 🔧 **Flexible Configuration**
- **Custom Fields**: Support for organization-specific JIRA fields
- **Template System**: Configurable ticket templates and formats
- **Priority Rules**: Customizable priority mapping logic
- **Assignment Logic**: Smart assignee selection based on break types

## 🏗️ Architecture

```python
class JiraIssueAgent:
    """
    Agent for creating JIRA-compatible issues from reconciliation breaks.
    Converts breaks and resolution data into proper JIRA import format.
    """
    
    async def create_jira_issues(
        breaks: List[Dict[str, Any]],
        resolution_data: Optional[Dict[str, Any]] = None,
        jira_config: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None,
        timeout: float = 120.0
    ) -> AgentResult
```

## 🚀 Usage Examples

### **Basic JIRA Ticket Creation**
```python
from agents.jira_issue_agent import JiraIssueAgent

agent = JiraIssueAgent()

# Create tickets for breaks requiring manual review
result = await agent.create_jira_issues(
    breaks=manual_review_breaks,
    additional_context="Q1 2024 dividend reconciliation - high priority resolution needed"
)

if agent.validate_jira_issues(result):
    jira_data = result.response.structured_data
    tickets = jira_data['jira_issues']
    print(f"Created {len(tickets)} JIRA tickets")
```

### **Advanced Usage with Configuration**
```python
# Custom JIRA configuration
jira_config = {
    "project_key": "RECON",
    "default_assignee": "reconciliation-team",
    "component": "Dividend Reconciliation",
    "labels": ["reconciliation", "dividend", "breaks", "urgent"],
    "epic_link": "RECON-100",
    "custom_fields": {
        "reconciliation_date": "2024-01-15",
        "system_source": "NBIM-Custody",
        "business_priority": "High"
    }
}

# Enhanced ticket creation with resolution context
result = await agent.create_jira_issues(
    breaks=complex_breaks,
    resolution_data=resolution_analysis,
    jira_config=jira_config,
    additional_context=detailed_context,
    timeout=120.0
)
```

### **Processing Results**
```python
if agent.validate_jira_issues(result):
    jira_data = result.response.structured_data
    
    # Get ticket statistics
    total_tickets = jira_data['total_issues_created']
    priority_breakdown = jira_data['issues_by_priority']
    
    # Export for JIRA import
    issues_csv = convert_to_csv(jira_data['jira_issues'])
    save_for_import(issues_csv, 'jira_import.csv')
    
    print(f"Created {total_tickets} tickets:")
    for priority, count in priority_breakdown.items():
        print(f"  - {priority}: {count}")
```

## 📊 Input/Output Formats

### **Input: Break Data**
```json
{
    "breaks": [
        {
            "break_id": "BRK_001",
            "description": "Tax amount mismatch between systems",
            "classification": "Value Discrepancy", 
            "severity": "High",
            "composite_key": "EVT001|US0378331005|2046251|ACC12345",
            "affected_field": "withholding_tax",
            "nbim_value": 360.00,
            "custody_value": 300.00,
            "impact_assessment": "Incorrect net settlement amount",
            "confidence": 0.95
        }
    ],
    "jira_config": {
        "project_key": "RECON",
        "component": "Dividend Reconciliation",
        "default_assignee": "recon-team"
    }
}
```

### **Output: JIRA Tickets**
```json
{
    "success": true,
    "jira_issues": [
        {
            "issue_key": "RECON-DIV-001",
            "summary": "[HIGH] Value Discrepancy - Withholding Tax Mismatch", 
            "description": "## Break Analysis\n**Break ID:** BRK_001\n**Classification:** Value Discrepancy...",
            "issue_type": "Bug",
            "priority": "High",
            "assignee": "recon-team",
            "reporter": "reconciliation-system",
            "component": "Dividend Reconciliation",
            "labels": ["reconciliation", "dividend", "tax-issue"],
            "custom_fields": {
                "break_id": "BRK_001",
                "composite_key": "EVT001|US0378331005|2046251|ACC12345",
                "affected_system": "Custody"
            }
        }
    ],
    "total_issues_created": 1,
    "issues_by_priority": {
        "High": 1
    },
    "issues_by_type": {
        "Bug": 1
    }
}
```

## 🎫 JIRA Ticket Structure

### **Ticket Summary Format**
```
[{PRIORITY}] {CLASSIFICATION} - {BRIEF_DESCRIPTION}

Examples:
- [HIGH] Value Discrepancy - Withholding Tax Amount Mismatch
- [CRITICAL] Missing Record - Dividend Payment Not Found in Custody
- [MEDIUM] Date Mismatch - Ex-Date Discrepancy Between Systems
```

### **Ticket Description Template**
```markdown
## Break Analysis
**Break ID:** BRK_001
**Classification:** Value Discrepancy  
**Severity:** High
**Confidence:** 95%

**Issue Description:**
Tax amount mismatch between NBIM and Custody systems for Apple Inc dividend payment.

**Values Comparison:**
- **NBIM Value:** 360.00
- **Custody Value:** 300.00
- **Difference:** 60.00

## Impact Assessment
Net settlement amount will be incorrect, affecting cash reconciliation and reporting.

## Required Action
Investigation required to:
1. Verify correct withholding tax rate for US securities (expected: 15%)
2. Identify which system has incorrect rate
3. Update affected system and recalculate settlement amount
4. Confirm no other securities affected by same issue

## Context
**Record Identifier:** EVT001|US0378331005|2046251|ACC12345
**Analysis Date:** 2024-01-15 14:30:00
**System:** NBIM-Custody Reconciliation
**Reconciliation Period:** Q1 2024
```

## 🎯 Priority Mapping

### **Severity to Priority Conversion**

| **Break Severity** | **JIRA Priority** | **SLA Target** | **Assignment** |
|-------------------|-------------------|----------------|----------------|
| `Critical` | Highest | 2 hours | Senior Team Lead |
| `High` | High | 4 hours | Team Lead |
| `Medium` | Medium | 1 business day | Any Team Member |
| `Low` | Low | 3 business days | Junior Analyst |

### **Custom Priority Rules**
```python
# Define custom priority mapping
priority_rules = {
    "missing_records_critical": {
        "condition": lambda b: b['classification'] == 'Missing Record' and b['severity'] == 'Critical',
        "priority": "Highest",
        "assignee": "senior-recon-lead"
    },
    "tax_errors_high": {
        "condition": lambda b: 'tax' in b['affected_field'].lower(),
        "priority": "High", 
        "assignee": "tax-specialist"
    }
}
```

## ⚙️ Configuration Options

### **JIRA Project Settings**
```python
jira_config = {
    # Required fields
    "project_key": "RECON",                    # JIRA project key
    "default_assignee": "reconciliation-team", # Default assignee
    
    # Optional fields  
    "component": "Dividend Reconciliation",    # Component name
    "default_priority": "Medium",              # Fallback priority
    "epic_link": "RECON-100",                 # Link to epic
    
    # Labels and categorization
    "labels": ["reconciliation", "dividend", "automated"],
    "fix_version": "2024.1",
    
    # Custom fields
    "custom_fields": {
        "reconciliation_date": "2024-01-15",
        "source_system": "NBIM-Custody", 
        "business_unit": "Investment Operations"
    }
}
```

### **Ticket Template Customization**
```python
# Custom ticket description template
custom_template = """
## Reconciliation Break Alert

**Break Details:**
- ID: {break_id}
- Type: {classification}
- Severity: {severity}

**Financial Impact:**
- Field: {affected_field}
- NBIM: {nbim_value}
- Custody: {custody_value}
- Difference: {difference}

**Investigation Required:**
{investigation_steps}

**Resolution Timeline:** {sla_target}
"""

# Use custom template
result = await agent.create_jira_issues(
    breaks=breaks,
    jira_config=jira_config,
    ticket_template=custom_template
)
```

## 🧪 Testing

### **Unit Tests**
```bash
python -m pytest agents/__tests__/test_jira_agent.py -v
```

### **Test Scenarios**
- ✅ **Basic Ticket Creation**: Standard break to ticket conversion
- ✅ **Priority Mapping**: Severity to priority conversion accuracy
- ✅ **Custom Fields**: Custom field population and formatting
- ✅ **Batch Processing**: Multiple breaks to multiple tickets
- ✅ **Error Handling**: Invalid data and API failures
- ✅ **Template Rendering**: Custom template formatting

### **Sample Test Cases**
```python
test_scenarios = [
    {
        "name": "high_priority_tax_issue",
        "break_data": {
            "classification": "Tax Calculation Error",
            "severity": "High",
            "affected_field": "withholding_tax"
        },
        "expected_priority": "High",
        "expected_assignee": "tax-specialist",
        "expected_labels": ["tax-issue", "high-priority"]
    },
    {
        "name": "missing_record_critical", 
        "break_data": {
            "classification": "Missing Record",
            "severity": "Critical"
        },
        "expected_priority": "Highest",
        "expected_summary_contains": "[CRITICAL] Missing Record"
    }
]
```

## 📈 Performance Characteristics

| **Capability** | **Description** |
|----------------|-----------------|
| Ticket Creation | Rich, structured JIRA tickets with business context |
| Template System | Flexible ticket formatting and customization |
| Batch Processing | Efficient handling of multiple tickets simultaneously |
| Integration Support | CSV export and direct API integration ready |

## 🔍 Advanced Features

### **Smart Assignee Selection**
```python
# Rule-based assignee selection
assignee_rules = {
    "tax_issues": {
        "condition": lambda b: 'tax' in b.get('affected_field', '').lower(),
        "assignee": "tax-specialist@company.com"
    },
    "missing_records": {
        "condition": lambda b: b.get('classification') == 'Missing Record',
        "assignee": "data-analyst@company.com" 
    },
    "high_value_breaks": {
        "condition": lambda b: abs(float(b.get('difference', 0))) > 10000,
        "assignee": "senior-analyst@company.com"
    }
}
```

### **Batch Import Support**
```python
# Generate CSV for JIRA batch import
def export_jira_csv(jira_issues):
    """Export tickets in JIRA CSV import format."""
    csv_fields = [
        'Summary', 'Issue Type', 'Priority', 'Assignee',
        'Reporter', 'Description', 'Component', 'Labels',
        'Custom Field 1', 'Custom Field 2'
    ]
    
    csv_data = []
    for issue in jira_issues:
        csv_row = {
            'Summary': issue['summary'],
            'Issue Type': issue['issue_type'],
            'Priority': issue['priority'],
            'Assignee': issue['assignee'],
            'Description': issue['description'],
            'Component': issue['component'],
            'Labels': ','.join(issue['labels']),
            'Custom Field 1': issue['custom_fields'].get('break_id'),
            'Custom Field 2': issue['custom_fields'].get('composite_key')
        }
        csv_data.append(csv_row)
    
    return pd.DataFrame(csv_data).to_csv(index=False)
```

### **Integration with Resolution Data**
```python
# Enhanced tickets with resolution context
result = await agent.create_jira_issues(
    breaks=manual_review_breaks,
    resolution_data={
        "resolutions": [
            {
                "break_id": "BRK_001",
                "corrected_value": "Manual review required",
                "reasoning": "Complex calculation requires specialist review"
            }
        ]
    },
    additional_context="Include resolution analysis in ticket descriptions"
)

# Results in enhanced ticket descriptions with resolution context
```

## ⚠️ Limitations & Best Practices

### **Known Limitations**
- **Template Constraints**: Limited by JIRA field character limits
- **Custom Fields**: Depends on target JIRA configuration
- **Bulk Creation**: API rate limits may affect large batches
- **Rich Formatting**: Limited rich text support in some JIRA versions

### **Best Practices**
- ✅ **Consistent Naming**: Use standardized ticket summaries and labels
- ✅ **Meaningful Descriptions**: Include all context needed for investigation
- ✅ **Appropriate Priority**: Map priorities to actual business impact
- ✅ **Clear Actions**: Provide specific investigation steps
- ✅ **Audit Trail**: Include all identifiers for traceability

### **Security Considerations**
```python
# Sanitize sensitive data before ticket creation
def sanitize_ticket_data(ticket_data):
    """Remove sensitive information from ticket content."""
    sensitive_patterns = [
        r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b',  # Credit card numbers
        r'\b\d{3}-\d{2}-\d{4}\b',                     # SSN patterns
        r'password[:\s=]+\w+',                        # Password fields
    ]
    
    for pattern in sensitive_patterns:
        ticket_data['description'] = re.sub(pattern, '[REDACTED]', ticket_data['description'])
    
    return ticket_data
```

## 🔄 Integration Examples

### **Streamlit Integration**
```python
# Used in breaks_streamlit_integration.py
def create_jira_tickets_for_rejected_fixes(rejected_fixes, resolution_lookup):
    # Create JIRA agent and generate tickets
    jira_agent = JiraIssueAgent()
    
    result = asyncio.run(jira_agent.create_jira_issues(
        breaks=breaks_for_jira,
        resolution_data=resolution_data,
        jira_config=user_jira_config,
        timeout=120.0
    ))
    
    if jira_agent.validate_jira_issues(result):
        display_jira_download_option(result.response.structured_data)
        return True
    return False
```

### **Production Workflow Integration**
```python
# Automated reconciliation pipeline
async def process_reconciliation_breaks():
    # 1. Identify breaks
    breaks = await identify_breaks(nbim_data, custody_data)
    
    # 2. Generate resolution suggestions
    resolutions = await generate_resolutions(breaks)
    
    # 3. Separate automated vs manual items
    manual_items = [b for b in breaks if needs_manual_review(b)]
    
    # 4. Create JIRA tickets for manual items
    if manual_items:
        jira_result = await jira_agent.create_jira_issues(
            breaks=manual_items,
            resolution_data=resolutions,
            jira_config=production_jira_config
        )
        
        # 5. Import tickets to JIRA
        if jira_agent.validate_jira_issues(jira_result):
            csv_content = export_jira_csv(jira_result.response.structured_data['jira_issues'])
            import_to_jira(csv_content)
```

---

## 🎯 Summary

The JIRA Issue Agent seamlessly bridges the gap between AI-identified reconciliation breaks and human investigation workflows. By generating rich, contextual JIRA tickets with proper prioritization and clear action items, it ensures that manual review cases are handled efficiently by reconciliation teams with all necessary information at their fingertips.

**Perfect for**: Reconciliation workflows, audit processes, data quality management, and any scenario requiring structured handoff from automated analysis to human investigation teams.