# 🔧 Breaks Resolution Agent

> **AI-powered fix suggestions and root cause analysis for reconciliation breaks**

## 📋 Overview

The Breaks Resolution Agent analyzes identified reconciliation breaks and provides specific, actionable fix suggestions. It focuses on delivering **exact corrected values** rather than vague recommendations, enabling automated remediation for high-confidence fixes while routing complex cases for manual review.

## ✨ Key Features

### 🎯 **Exact Value Corrections**
- **Precise Fixes**: Provides exact corrected numbers, dates, or values
- **Calculation-Based**: Shows mathematical reasoning for corrections
- **Confidence Scoring**: Self-assessment of fix reliability (0-1 scale)
- **Manual Review Routing**: Identifies cases requiring human investigation

### 🧠 **Intelligent Analysis**
- **Root Cause Identification**: Determines underlying causes of breaks
- **Pattern Recognition**: Identifies systemic issues across multiple breaks  
- **Historical Context**: Leverages previous reconciliation patterns
- **Business Logic**: Applies financial domain knowledge (tax rates, settlement rules)

### 🤖 **Automation Support**
- **High-Confidence Automation**: Fixes with >80% confidence can be automated
- **Risk Assessment**: Conservative approach for financial data
- **Audit Trail**: Complete reasoning for each suggested fix
- **Rollback Support**: Original values preserved for safety

## 🏗️ Architecture

```python
class BreaksResolutionAgent:
    """
    Agent for analyzing break root causes and suggesting resolutions.
    Takes breaks from the breaks identifier and provides actionable fixes.
    """
    
    async def analyze_and_resolve(
        breaks: List[Dict[str, Any]],
        additional_context: Optional[str] = None,
        historical_patterns: Optional[Dict[str, Any]] = None,
        timeout: float = 180.0,
        temperature: float = 0.1
    ) -> AgentResult
```

## 🚀 Usage Examples

### **Basic Resolution Analysis**
```python
from agents.breaks_resolution_agent import BreaksResolutionAgent

agent = BreaksResolutionAgent()

# Analyze breaks and get fix suggestions
result = await agent.analyze_and_resolve(
    breaks=identified_breaks,
    additional_context="Q1 2024 dividend reconciliation, US tax rate 15%"
)

if agent.validate_resolution(result):
    resolutions = result.response.structured_data['resolutions']
    for resolution in resolutions:
        break_id = resolution['break_id']
        suggested_fix = resolution['corrected_value']
        confidence = resolution['confidence']
        print(f"{break_id}: {suggested_fix} (confidence: {confidence:.0%})")
```

### **Advanced Usage with Historical Patterns**
```python
# Enhanced analysis with historical context
historical_patterns = {
    "common_issues": [
        "Tax rate mismatches for US securities",
        "Timing differences for ex-date near month-end"
    ],
    "previous_fixes": {
        "tax_rate_issues": "Usually Custody system needs update",
        "timing_differences": "Usually resolve within 2 days"
    }
}

result = await agent.analyze_and_resolve(
    breaks=complex_breaks,
    additional_context=business_context,
    historical_patterns=historical_patterns,
    timeout=120.0
)
```

### **Processing Results**
```python
# Extract different types of fixes
automatable_fixes = agent.get_fixes_with_values(result)
manual_review_items = agent.get_manual_review_fixes(result) 
high_confidence_fixes = agent.get_high_confidence_fixes(result, min_confidence=0.8)

print(f"Automatable: {len(automatable_fixes)}")
print(f"Manual review: {len(manual_review_items)}")
print(f"High confidence: {len(high_confidence_fixes)}")
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
            "difference": 60.00,
            "impact_assessment": "Net settlement amount will be incorrect",
            "confidence": 0.95
        }
    ]
}
```

### **Output: Resolution Suggestions**
```json
{
    "success": true,
    "resolutions": [
        {
            "break_id": "BRK_001", 
            "corrected_value": "360.00",
            "reasoning": "US securities have 15% withholding tax rate. Correct tax = gross_amount * 0.15 = 2400 * 0.15 = 360.00. Custody system appears to have incorrect rate.",
            "confidence": 0.95
        },
        {
            "break_id": "BRK_002",
            "corrected_value": "Manual review required", 
            "reasoning": "Missing record requires investigation of source systems to determine if dividend was actually paid",
            "confidence": 0.0
        }
    ],
    "total_breaks_analyzed": 2,
    "total_resolvable": 1,
    "total_requiring_manual_review": 1,
    "automation_potential": {
        "fully_automatable": 1,
        "partially_automatable": 0,
        "manual_only": 1
    }
}
```

## 🎯 Resolution Types & Examples

### **Exact Value Corrections**
```json
{
    "break_id": "TAX_001",
    "corrected_value": "450.00", 
    "reasoning": "15% withholding tax on $3000 gross = $450.00",
    "confidence": 0.95
}
```

### **Date Corrections**
```json
{
    "break_id": "DATE_001", 
    "corrected_value": "2024-02-15",
    "reasoning": "Pay date should be T+2 business days after ex-date 2024-02-13", 
    "confidence": 0.88
}
```

### **Quantity Corrections**
```json
{
    "break_id": "QTY_001",
    "corrected_value": "2500",
    "reasoning": "Post 2:1 stock split, original 1250 shares becomes 2500",
    "confidence": 0.92
}
```

### **Manual Review Cases**
```json
{
    "break_id": "MISSING_001",
    "corrected_value": "Manual review required", 
    "reasoning": "Record missing in Custody system - need to verify if dividend payment occurred",
    "confidence": 0.0
}
```

## 🔍 Resolution Categories

### **Automation Potential Classification**

| **Category** | **Criteria** | **Action** |
|--------------|--------------|------------|
| `fully_automatable` | Confidence ≥ 0.8 AND specific value provided | Apply fix automatically |
| `partially_automatable` | Confidence 0.5-0.8 AND specific value | Suggest with approval |
| `manual_only` | Confidence < 0.5 OR "Manual review required" | Route to investigation team |

### **Common Fix Types**

| **Fix Type** | **Description** | **Example** |
|--------------|-----------------|-------------|
| **Tax Calculation** | Standard withholding rates | 15% for US securities |
| **Currency Conversion** | FX rate corrections | USD to EUR at 1.0850 |
| **Date Standardization** | Business day calculations | T+2 settlement dates |
| **Quantity Adjustments** | Corporate action impacts | Stock splits, spin-offs |
| **Reference Data** | Security identifier fixes | ISIN/SEDOL corrections |

## ⚙️ Configuration

### **Analysis Parameters**
```python
result = await agent.analyze_and_resolve(
    breaks=break_list,
    additional_context=context,
    temperature=0.1,        # Low for consistent reasoning
    timeout=120.0,          # Reduced timeout for simpler analysis  
    historical_patterns=patterns
)
```

### **Confidence Thresholds**
```python
# Get only high-confidence fixes
high_conf_fixes = agent.get_high_confidence_fixes(
    result, 
    min_confidence=0.8  # Only fixes with >80% confidence
)

# Filter by automation potential
automatable = [r for r in resolutions 
               if r.get('confidence', 0) >= 0.8 
               and r.get('corrected_value') != "Manual review required"]
```

## 🧪 Testing

### **Unit Tests**
```bash
python -m pytest agents/__tests__/test_breaks_resolution_agent.py -v
```

### **Test Scenarios**
- ✅ **Tax Calculation Fixes**: Standard withholding rates
- ✅ **Value Corrections**: Mathematical error fixes  
- ✅ **Date Adjustments**: Business day calculations
- ✅ **Missing Record Cases**: Manual review routing
- ✅ **Complex Breaks**: Multi-factor resolution analysis
- ✅ **Edge Cases**: Confidence boundary testing

### **Sample Test Cases**
```python
test_cases = [
    {
        "break_type": "tax_error",
        "input": {"gross": 1000, "tax_nbim": 150, "tax_custody": 100},
        "expected_fix": "150.00", 
        "expected_confidence": 0.95,
        "reasoning_contains": "15% withholding tax"
    },
    {
        "break_type": "missing_record", 
        "input": {"nbim_exists": True, "custody_exists": False},
        "expected_fix": "Manual review required",
        "expected_confidence": 0.0,
        "reasoning_contains": "investigation"
    }
]
```

## 📈 Performance Characteristics

| **Capability** | **Description** |
|----------------|-----------------|
| Fix Accuracy | High-quality exact value corrections with detailed reasoning |
| Automation Support | Confidence-based automation with conservative thresholds |
| Processing Speed | Efficient analysis of multiple breaks simultaneously |
| Cost Optimization | Streamlined prompts for cost-effective operations |

## 🔍 Helper Methods

### **Result Processing**
```python
# Get fixes that provide specific values  
value_fixes = agent.get_fixes_with_values(result)

# Get items requiring manual review
manual_items = agent.get_manual_review_fixes(result)

# Get high-confidence automatable fixes
auto_fixes = agent.get_high_confidence_fixes(result, min_confidence=0.8)

# Generate fix script for automation
for fix in auto_fixes:
    script = agent.generate_fix_script(fix)
    print(script)
```

### **Validation & Quality Checks**
```python
# Validate resolution results
is_valid = agent.validate_resolution(result)

if is_valid:
    resolutions = result.response.structured_data['resolutions']
    for resolution in resolutions:
        break_id = resolution['break_id']
        has_specific_fix = resolution['corrected_value'] != "Manual review required"
        confidence = resolution['confidence']
        
        print(f"{break_id}: {'Automatable' if has_specific_fix and confidence > 0.8 else 'Manual'}")
```

## 💡 Advanced Features

### **Fix Script Generation**
```python
# Generate Python script for applying fixes
fix_script = agent.generate_fix_script(resolution)

# Example generated script:
"""
def apply_fix(data, break_id, field_name):
    corrected_value = "360.00"
    
    for record in data:
        if record.get("break_id") == break_id:
            record[field_name] = corrected_value
            print(f"Updated {field_name} to {corrected_value} for break {break_id}")
            return True
    return False
"""
```

### **Historical Pattern Learning**
```python
# Build patterns from previous reconciliations
historical_patterns = {
    "common_issues": [
        "Tax rate mismatches for US securities",
        "Timing differences for ex-date near month-end",
        "Quantity discrepancies for stock splits"
    ],
    "previous_fixes": {
        "tax_rate_issues": "Usually Custody system needs update",
        "timing_differences": "Usually resolve within 2 days"
    },
    "success_rates": {
        "tax_corrections": 0.95,
        "date_corrections": 0.88,
        "manual_reviews": 0.75
    }
}
```

## ⚠️ Limitations & Best Practices

### **Known Limitations**
- **Complex Calculations**: Limited to standard financial calculations
- **External Data**: Cannot access real-time market data or external systems
- **Multi-Step Fixes**: Each resolution addresses single break only
- **Context Dependency**: Quality depends on provided business context

### **Best Practices**
- ✅ **Conservative Automation**: Use high confidence thresholds (≥0.8) for automation
- ✅ **Audit Trail**: Always log original values and applied fixes
- ✅ **Manual Review**: Route uncertain cases to human experts
- ✅ **Testing**: Validate fixes on test data before production application
- ✅ **Monitoring**: Track fix success rates and adjust thresholds

### **Risk Mitigation**
```python
# Conservative automation approach
AUTOMATION_CONFIDENCE_THRESHOLD = 0.8
MAX_AUTOMATED_FIXES_PER_RUN = 50
REQUIRE_APPROVAL_FOR_LARGE_AMOUNTS = True

for fix in resolutions:
    if (fix['confidence'] >= AUTOMATION_CONFIDENCE_THRESHOLD and
        fix['corrected_value'] != "Manual review required"):
        
        # Additional safety checks
        if requires_approval(fix):
            route_for_approval(fix)
        else:
            apply_fix_automatically(fix)
    else:
        route_for_manual_review(fix)
```

## 🔄 Integration Examples

### **Streamlit Integration**
```python
# Used in breaks_streamlit_integration.py
with st.spinner("🤖 AI is analyzing breaks for resolution strategies..."):
    result = asyncio.run(agent.analyze_and_resolve(
        breaks=identified_breaks,
        additional_context=business_context,
        timeout=90.0
    ))

if agent.validate_resolution(result):
    display_resolution_table(breaks_data, result.response.structured_data)
```

### **Automated Pipeline**
```python
# Production reconciliation pipeline
def process_daily_reconciliation():
    # 1. Identify breaks
    breaks_result = await breaks_agent.identify_breaks(nbim_data, custody_data)
    
    # 2. Generate fix suggestions  
    resolution_result = await resolution_agent.analyze_and_resolve(
        breaks=breaks_result.response.structured_data['breaks']
    )
    
    # 3. Apply high-confidence fixes automatically
    auto_fixes = resolution_agent.get_high_confidence_fixes(
        resolution_result, min_confidence=0.8
    )
    
    applied_fixes = apply_fixes_to_data(auto_fixes)
    
    # 4. Route remaining items for manual review
    manual_items = resolution_agent.get_manual_review_fixes(resolution_result)
    create_jira_tickets(manual_items)
    
    return {
        'automated_fixes': len(applied_fixes),
        'manual_review_items': len(manual_items)
    }
```

---

## 🎯 Summary

The Breaks Resolution Agent transforms break analysis from manual investigation to intelligent, automated resolution. By providing exact corrected values with detailed reasoning and confidence scoring, it enables safe automation of routine fixes while ensuring complex cases receive appropriate human attention.

**Perfect for**: Automated reconciliation workflows, financial data remediation, audit processes, and any scenario requiring intelligent break resolution with appropriate risk management.