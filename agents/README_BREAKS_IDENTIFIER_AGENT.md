# 🔍 Breaks Identifier Agent

> **Pure LLM-driven break identification and classification for financial data reconciliation**

## 📋 Overview

The Breaks Identifier Agent uses Large Language Models to dynamically identify and classify discrepancies between NBIM and Custody dividend data. Unlike traditional rule-based systems, it leverages AI to understand business context and identify breaks without hardcoded categories, making it adaptable to new types of discrepancies.

## ✨ Key Features

### 🧠 **Dynamic Break Detection**
- **LLM-Driven Analysis**: No hardcoded business rules or break categories
- **Contextual Understanding**: Analyzes data within business domain context
- **Composite Key Matching**: Uses multi-field keys for robust record linking
- **Adaptive Classification**: Discovers new break types automatically

### 🔑 **Advanced Record Matching**
- **Composite Keys**: Links records using coac_event_key + ISIN + SEDOL + bank_account
- **Flexible Matching**: Handles case variations and formatting differences  
- **Unique Identification**: Each break tied to specific composite key
- **Missing Record Detection**: Identifies records present in one dataset but not the other

### 📊 **Intelligent Classification**
- **Severity Assessment**: Automatically assigns impact-based severity levels
- **Confidence Scoring**: Self-assessment of detection reliability (0-1 scale)
- **Impact Analysis**: Business impact evaluation for each break
- **Pattern Recognition**: Identifies systemic issues across multiple records

## 🏗️ Architecture

```python
class BreaksIdentifierAgent:
    """
    Pure LLM-driven agent for identifying breaks in reconciliation.
    No hardcoded categories - lets the LLM determine what's important.
    Uses composite keys for unique record identification.
    """
    
    async def identify_breaks(
        nbim_data: List[Dict[str, Any]],
        custody_data: List[Dict[str, Any]], 
        mappings: Optional[Dict[str, str]] = None,
        primary_keys: Optional[List[str]] = None,
        additional_context: Optional[str] = None
    ) -> AgentResult
```

## 🚀 Usage Examples

### **Basic Break Identification**
```python
from agents.breaks_identifier_agent import BreaksIdentifierAgent

agent = BreaksIdentifierAgent()

# Simple break detection
result = await agent.identify_breaks(
    nbim_data=nbim_records,
    custody_data=custody_records,
    mappings={'isin': 'isin_code', 'gross_amount': 'gross_dividend'}
)

if agent.validate_breaks(result):
    breaks = result.response.structured_data['breaks']
    print(f"Found {len(breaks)} breaks requiring attention")
```

### **Advanced Usage with Context**
```python
# Enhanced analysis with business context
context = """
Quarterly dividend reconciliation for Q1 2024.
Standard US withholding tax rate is 15%.
Settlement typically T+2 for US securities.
Focus on tax calculation accuracy and missing payments.
"""

result = await agent.identify_breaks(
    nbim_data=nbim_records,
    custody_data=custody_records, 
    mappings=field_mappings,
    additional_context=context,
    timeout=180.0
)

# Analyze results
breaks_data = result.response.structured_data
total_breaks = breaks_data['total_breaks_found']
classifications = breaks_data['classification_summary']
severities = breaks_data['severity_summary']
```

### **Batch Processing for Large Datasets**
```python
# Handle large datasets efficiently
result = await agent.identify_breaks_in_batches(
    nbim_data=large_nbim_dataset,
    custody_data=large_custody_dataset,
    mappings=field_mappings,
    batch_size=100,  # Process 100 records at a time
    additional_context=business_context
)
```

## 📊 Input/Output Formats

### **Input Data Structure**
```json
{
    "nbim_data": [
        {
            "coac_event_key": "EVT001",
            "isin": "US0378331005", 
            "sedol": "2046251",
            "bank_account": "ACC12345",
            "gross_amount": 2400.00,
            "withholding_tax": 360.00,
            "net_amount": 2040.00
        }
    ],
    "custody_data": [
        {
            "event_id": "EVT001",
            "isin_code": "US0378331005",
            "sedol_code": "2046251", 
            "account_number": "ACC12345",
            "gross_dividend": 2400.00,
            "tax_withheld": 300.00,
            "net_dividend": 2100.00
        }
    ],
    "mappings": {
        "coac_event_key": "event_id",
        "withholding_tax": "tax_withheld"
    }
}
```

### **Output Structure**
```json
{
    "success": true,
    "breaks": [
        {
            "break_id": "BRK_001",
            "description": "Tax amount mismatch between systems",
            "classification": "Value Discrepancy",
            "severity": "High",
            "composite_key": "EVT001|US0378331005|2046251|ACC12345",
            "composite_key_components": {
                "coac_event_key": "EVT001",
                "isin": "US0378331005",
                "sedol": "2046251", 
                "bank_account": "ACC12345"
            },
            "affected_field": "withholding_tax",
            "nbim_value": 360.00,
            "custody_value": 300.00,
            "difference": 60.00,
            "impact_assessment": "Incorrect net settlement amount",
            "confidence": 0.95
        }
    ],
    "total_breaks_found": 1,
    "classification_summary": {
        "Value Discrepancy": 1
    },
    "severity_summary": {
        "High": 1
    },
    "overall_assessment": "Found tax calculation discrepancies requiring investigation"
}
```

## 🔍 Break Classification Types

### **Dynamic Categories** (LLM-Determined)
The agent dynamically discovers break types based on the data. Common categories include:

| **Classification** | **Description** | **Typical Severity** |
|-------------------|-----------------|---------------------|
| `Value Discrepancy` | Numerical differences in amounts | Medium-High |
| `Missing Record` | Record exists in one system only | Critical |
| `Date Mismatch` | Different dates between systems | Low-Medium |
| `Tax Calculation Error` | Incorrect withholding calculations | High |
| `Currency Inconsistency` | Currency code differences | Medium |
| `Quantity Variance` | Share quantity mismatches | High |
| `Reference Data Error` | ISIN/SEDOL mismatches | Medium |

### **Severity Levels** (LLM-Assigned)
- **Critical**: Complete record missing, major calculation errors
- **High**: Significant value differences, tax errors
- **Medium**: Minor discrepancies, reference data issues  
- **Low**: Formatting differences, minor date variations

## 🔑 Composite Key System

### **Primary Key Fields**
```python
primary_key_fields = [
    'coac_event_key',  # Event identifier
    'isin',            # Security identifier
    'sedol',           # Alternative security ID
    'bank_account'     # Account identifier  
]
```

### **Key Building Logic**
```python
# Composite key: "EVT001|US0378331005|2046251|ACC12345"
composite_key = '|'.join([
    str(record.get('coac_event_key', 'NULL')),
    str(record.get('isin', 'NULL')),
    str(record.get('sedol', 'NULL')), 
    str(record.get('bank_account', 'NULL'))
])
```

### **Benefits**
- ✅ **Unique Identification**: Each record has unique composite identifier
- ✅ **Robust Matching**: Handles partial key matches
- ✅ **Break Traceability**: Each break linked to specific record combination
- ✅ **Flexible Matching**: Case-insensitive and format-tolerant

## ⚙️ Configuration Options

### **Analysis Parameters**
```python
result = await agent.identify_breaks(
    nbim_data=data1,
    custody_data=data2,
    temperature=0.1,           # Low for consistent analysis
    timeout=180.0,             # API timeout in seconds
    primary_keys=custom_keys,  # Override default composite key fields
    additional_context=context # Business context for analysis
)
```

### **Batch Processing Settings**
```python
result = await agent.identify_breaks_in_batches(
    nbim_data=large_dataset1,
    custody_data=large_dataset2,
    batch_size=100,           # Records per batch
    mappings=field_mappings,
    additional_context=context
)
```

## 🧪 Testing

### **Unit Tests**
```bash
python -m pytest agents/__tests__/test_breaks_identifier.py -v
```

### **Test Scenarios**
- ✅ **Value Mismatches**: Different amounts between systems
- ✅ **Missing Records**: Records in one dataset only
- ✅ **Tax Calculations**: Withholding tax accuracy
- ✅ **Date Discrepancies**: Ex-date and pay-date variations
- ✅ **Composite Key Matching**: Multi-field record linking
- ✅ **Edge Cases**: Null values, formatting differences

### **Sample Test Data**
```python
# Test case: Tax calculation error
nbim_record = {
    'coac_event_key': 'EVT001',
    'isin': 'US0378331005',
    'gross_amount': 1000.00,
    'withholding_tax': 150.00  # 15% tax
}

custody_record = {
    'event_id': 'EVT001', 
    'isin_code': 'US0378331005',
    'gross_dividend': 1000.00,
    'tax_withheld': 100.00     # Incorrect 10% tax
}

# Expected: Break detected with "Tax Calculation Error" classification
```

## 📈 Performance Characteristics

| **Capability** | **Description** |
|----------------|-----------------|
| Break Detection | Dynamic LLM-powered classification without hardcoded rules |
| Processing Scale | Efficient batch processing for large datasets |
| Composite Key Matching | Robust multi-field record linking system |
| Classification Depth | Discovers new break types automatically |

## 🔍 Analysis Results

### **Result Extraction Methods**
```python
# Get breaks by classification
value_breaks = agent.get_breaks_by_classification(result, "Value Discrepancy")

# Get breaks by severity 
critical_breaks = agent.get_breaks_by_severity(result, "Critical")

# Get breaks for specific composite key
key_breaks = agent.get_breaks_by_composite_key(result, "EVT001|US123|SEDOL123|ACC001")

# Group all breaks by composite key
grouped_breaks = agent.group_breaks_by_composite_key(result)

# Extract unique classifications discovered
classifications, severities = agent.extract_unique_classifications(result)
```

### **Validation & Quality Checks**
```python
# Validate results
is_valid = agent.validate_breaks(result)

# Check result status
if result.status == AgentStatus.COMPLETED:
    breaks_data = result.response.structured_data
    total_breaks = breaks_data['total_breaks_found']
else:
    print(f"Analysis failed: {result.error}")
```

## ⚠️ Limitations & Considerations

### **Known Limitations**
- **Token Limits**: Large datasets may require batching
- **Context Dependency**: Analysis quality depends on business context provided
- **LLM Variability**: Small variations in classification names between runs
- **Cost Scaling**: API costs scale with dataset size

### **Best Practices**
- ✅ Provide rich business context in additional_context parameter
- ✅ Use appropriate batch sizes for large datasets (100-200 records)
- ✅ Set conservative timeout values for complex analyses
- ✅ Validate composite key fields exist in your data
- ✅ Review low-confidence breaks manually

### **Error Handling**
```python
try:
    result = await agent.identify_breaks(nbim_data, custody_data)
    if not agent.validate_breaks(result):
        print("Analysis validation failed")
except Exception as e:
    print(f"Break identification failed: {e}")
```

## 🔄 Integration Examples

### **Streamlit Integration**
```python
# Used in breaks_streamlit_integration.py
with st.spinner("🤖 AI is analyzing breaks..."):
    result = asyncio.run(agent.identify_breaks(
        nbim_data=nbim_records,
        custody_data=custody_records,
        mappings=field_mappings,
        timeout=180.0
    ))

if agent.validate_breaks(result):
    display_breaks_results(result.response.structured_data)
```

### **Batch Processing Pipeline**
```python
# Process multiple reconciliation files
for recon_date in date_range:
    nbim_file = f"nbim_{recon_date}.csv"
    custody_file = f"custody_{recon_date}.csv"
    
    result = await agent.identify_breaks(
        nbim_data=load_csv(nbim_file),
        custody_data=load_csv(custody_file),
        mappings=standard_mappings,
        additional_context=f"Daily reconciliation for {recon_date}"
    )
    
    save_breaks_report(result, f"breaks_{recon_date}.json")
```

## 🎯 Use Cases

### **Primary Use Cases**
- ✅ **Daily Reconciliation**: Automated break detection for routine processing
- ✅ **Month-End Closes**: Comprehensive analysis for period-end reporting
- ✅ **Data Migration**: Validation during system migrations
- ✅ **Audit Support**: Detailed break analysis for regulatory reviews

### **Advanced Applications**
- 🔄 **Pattern Analysis**: Identify recurring break types for process improvement
- 📊 **Trend Monitoring**: Track break frequency and types over time
- 🎯 **Root Cause Analysis**: Detailed investigation of systemic issues
- 🔍 **Data Quality Assessment**: Overall data integrity evaluation

---

## 🎯 Summary

The Breaks Identifier Agent revolutionizes financial reconciliation by bringing AI-powered intelligence to break detection. Its dynamic classification system, composite key matching, and contextual understanding make it adaptable to new scenarios while maintaining high accuracy and providing detailed insights for reconciliation teams.

**Perfect for**: Financial reconciliation, data quality analysis, audit processes, and any scenario requiring intelligent discrepancy detection between datasets.