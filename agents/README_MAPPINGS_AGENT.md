# 🗺️ Mappings Agent

> **Intelligent header mapping between different data formats using LLM-powered field analysis**

## 📋 Overview

The Mappings Agent uses Large Language Models to automatically identify and map corresponding fields between NBIM and Custody data files. It eliminates manual header mapping by analyzing field names, sample data, and business context to create accurate field mappings.

## ✨ Key Features

### 🧠 **Intelligent Field Analysis**
- **Semantic Understanding**: Analyzes field meaning beyond just names
- **Context Awareness**: Uses business domain knowledge for financial data
- **Sample Data Analysis**: Examines actual data values for better matching
- **Confidence Scoring**: Provides reliability scores for each mapping

### 🎯 **Smart Mapping Logic**
- **Fuzzy Matching**: Handles variations in naming conventions
- **Synonym Recognition**: Maps equivalent terms (e.g., "ISIN" ↔ "isin_code")
- **Pattern Detection**: Identifies common field patterns
- **Multi-factor Analysis**: Combines name similarity, data types, and business logic

## 🏗️ Architecture

```python
class MappingsAgent:
    """
    Agent for creating intelligent field mappings between datasets.
    Uses LLM to understand field semantics and create accurate mappings.
    """
    
    def create_mappings(
        source_headers: List[str],
        target_headers: List[str], 
        context: Optional[str] = None,
        sample_data: Optional[Dict] = None
    ) -> AgentResult
```

## 🚀 Usage Examples

### **Basic Usage**
```python
from agents.mappings_agent import MappingsAgent

agent = MappingsAgent()

# Simple header mapping
result = await agent.create_mappings(
    source_headers=['coac_event_key', 'isin', 'security_name'],
    target_headers=['event_id', 'isin_code', 'instrument_name'],
    context="Financial dividend reconciliation data"
)

if agent.validate_mappings(result):
    mappings = result.response.structured_data['mappings']
    for mapping in mappings:
        print(f"{mapping['source_header']} → {mapping['target_header']} ({mapping['confidence']:.0%})")
```

### **Advanced Usage with Sample Data**
```python
# Enhanced mapping with sample data analysis
sample_data = {
    'source_sample': {
        'coac_event_key': ['EVT001', 'EVT002'], 
        'gross_amount': [1000.50, 2500.75]
    },
    'target_sample': {
        'event_id': ['EVT001', 'EVT002'],
        'gross_dividend': [1000.50, 2500.75]
    }
}

result = await agent.create_mappings(
    source_headers=nbim_headers,
    target_headers=custody_headers,
    context="NBIM to Custody dividend reconciliation",
    sample_data=sample_data
)
```

## 📊 Input/Output Formats

### **Input**
```json
{
    "source_headers": ["coac_event_key", "isin", "gross_amount"],
    "target_headers": ["event_id", "isin_code", "gross_dividend"], 
    "context": "Dividend reconciliation between NBIM and Custody systems",
    "sample_data": {
        "source_sample": {...},
        "target_sample": {...}
    }
}
```

### **Output**
```json
{
    "success": true,
    "mappings": [
        {
            "source_header": "coac_event_key",
            "target_header": "event_id", 
            "confidence": 0.95,
            "reasoning": "Both fields contain event identifiers with matching patterns",
            "mapping_type": "exact_match"
        }
    ],
    "unmapped_headers": ["field_without_match"],
    "mapping_statistics": {
        "total_source_headers": 15,
        "total_target_headers": 18,
        "successful_mappings": 12,
        "high_confidence_mappings": 10
    }
}
```

## 🔍 Mapping Types

| **Type** | **Description** | **Confidence Range** |
|----------|-----------------|---------------------|
| `exact_match` | Perfect semantic match | 90-100% |
| `semantic_match` | Same meaning, different names | 75-90% |
| `pattern_match` | Similar data patterns/formats | 60-75% |
| `contextual_match` | Business context suggests mapping | 50-60% |
| `uncertain_match` | Possible but needs verification | <50% |

## ⚙️ Configuration

### **LLM Settings**
```python
agent = MappingsAgent()
result = await agent.create_mappings(
    source_headers=headers1,
    target_headers=headers2,
    temperature=0.1,           # Low for consistent results
    timeout=60.0,              # API timeout
    max_retries=3              # Error handling
)
```

### **Mapping Options**
```python
# Strict mapping (high confidence only)
result = await agent.create_mappings(
    source_headers=headers1,
    target_headers=headers2,
    min_confidence=0.8,        # Only high-confidence mappings
    allow_multiple_matches=False  # One-to-one mapping only
)
```

## 🧪 Testing

### **Unit Tests**
```bash
python -m pytest agents/__tests__/test_mappings_agent.py -v
```

### **Test Coverage**
- ✅ Basic field mapping scenarios  
- ✅ Complex financial field variations
- ✅ Edge cases (empty headers, duplicates)
- ✅ Error handling (API failures, malformed data)
- ✅ Performance with large header sets

### **Sample Test Data**
```python
# Common financial field variations tested
test_cases = [
    ("ISIN", "isin_code", 0.95),
    ("ex_date", "ex_dividend_date", 0.90), 
    ("gross_amount", "gross_dividend", 0.88),
    ("withholding_tax", "tax_withheld", 0.85)
]
```

## 🔧 Customization

### **Custom Business Logic**
```python
class CustomMappingsAgent(MappingsAgent):
    """Extended agent with custom business rules."""
    
    def _build_system_prompt(self) -> str:
        base_prompt = super()._build_system_prompt()
        custom_rules = """
        CUSTOM BUSINESS RULES:
        - 'coac_event_key' always maps to 'event_id'
        - Currency fields must match exactly
        - Date fields can have format variations
        """
        return base_prompt + custom_rules
```

### **Domain-Specific Mappings**
```python
# Add domain-specific context
financial_context = """
This is dividend reconciliation data containing:
- Security identifiers (ISIN, SEDOL, CUSIP)
- Financial amounts (gross, net, tax amounts)
- Dates (ex-date, pay-date, record-date)
- Quantities and rates
"""

result = await agent.create_mappings(
    source_headers=headers1,
    target_headers=headers2,
    context=financial_context
)
```

## 📈 Performance Characteristics

| **Capability** | **Description** |
|----------------|-----------------|
| Mapping Accuracy | High accuracy with semantic understanding |
| Processing Speed | Fast analysis of header relationships |
| API Cost | Optimized prompts for cost efficiency |
| Confidence Scoring | Reliable confidence assessment (0-1 scale) |

## ⚠️ Limitations & Considerations

### **Known Limitations**
- **Complex Transformations**: Cannot handle data transformations, only field mappings
- **One-to-Many**: Limited support for one field mapping to multiple targets
- **Context Dependency**: Performance varies with quality of business context provided
- **Language Support**: Optimized for English field names

### **Best Practices**
- ✅ Provide rich business context in prompts
- ✅ Include sample data when available
- ✅ Review low-confidence mappings manually
- ✅ Use validation functions to verify results
- ✅ Set appropriate confidence thresholds for automation

## 🔄 Integration Examples

### **Streamlit Integration**
```python
# Used in app.py for interactive mapping
if st.button("Generate AI Mappings"):
    result = await agent.create_mappings(
        source_headers=nbim_headers,
        target_headers=custody_headers, 
        context=user_context
    )
    display_mapping_results(result)
```

### **Batch Processing**
```python
# Process multiple file pairs
for nbim_file, custody_file in file_pairs:
    nbim_headers = get_headers(nbim_file)
    custody_headers = get_headers(custody_file)
    
    result = await agent.create_mappings(
        source_headers=nbim_headers,
        target_headers=custody_headers
    )
    
    save_mappings(result, f"mappings_{nbim_file}.json")
```

## 🛠️ Error Handling

### **Common Errors & Solutions**

| **Error** | **Cause** | **Solution** |
|-----------|-----------|-------------|
| `TimeoutError` | API timeout | Increase timeout or reduce header count |
| `ValidationError` | Invalid response format | Check LLM response schema |
| `EmptyMappingsError` | No mappings found | Provide better context or sample data |
| `ConfidenceError` | All mappings below threshold | Lower confidence threshold or review manually |

### **Error Recovery**
```python
try:
    result = await agent.create_mappings(headers1, headers2)
except TimeoutError:
    # Fallback to simpler matching
    result = await agent.create_mappings(
        headers1, headers2, 
        timeout=120.0,
        fallback_to_simple=True
    )
```

## 📋 Validation & Quality Checks

### **Built-in Validation**
```python
# Automatic validation
is_valid = agent.validate_mappings(result)

# Custom validation rules
def custom_validate(mappings):
    for mapping in mappings:
        # Ensure required fields are mapped
        if mapping['source_header'] in REQUIRED_FIELDS:
            assert mapping['confidence'] > 0.8
    return True
```

### **Quality Metrics**
- **Coverage**: Percentage of source fields mapped
- **Confidence**: Average confidence score
- **Accuracy**: Manual verification results
- **Completeness**: Critical fields mapped successfully

---

## 🎯 Summary

The Mappings Agent provides intelligent, automated field mapping capabilities that significantly reduce manual effort in data reconciliation workflows. By leveraging LLM understanding of business context and field semantics, it achieves high accuracy while providing transparency through confidence scoring and detailed reasoning.

**Perfect for**: Data integration projects, ETL processes, system migrations, and any scenario requiring intelligent field mapping between different data formats.