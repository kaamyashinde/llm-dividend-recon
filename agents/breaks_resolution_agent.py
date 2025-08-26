"""
Breaks Resolution Agent - Analyzes breaks to identify root causes and suggest fixes.

This agent takes identified breaks from the breaks identifier agent and:
1. Analyzes the root cause of each break
2. Suggests specific fixes that can be applied
3. Provides confidence levels for each suggested fix
"""

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import asyncio
import json
from enum import Enum

# Import utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import (
    openai_client,
    logger,
    ResponseParser,
    BaseAgentResponse,
    AgentExecutor,
    AgentResult,
    AgentStatus
)


# Define fix types
class FixType(str, Enum):
    """Types of fixes that can be applied."""
    DATA_CORRECTION = "data_correction"
    MAPPING_ADJUSTMENT = "mapping_adjustment"
    CALCULATION_RECALC = "calculation_recalculation"
    DATE_ALIGNMENT = "date_alignment"
    TAX_RATE_UPDATE = "tax_rate_update"
    FX_RATE_UPDATE = "fx_rate_update"
    MANUAL_REVIEW = "manual_review_required"
    SYSTEM_SYNC = "system_synchronization"
    REFERENCE_DATA_UPDATE = "reference_data_update"





class BreakResolution(BaseModel):
    """Simplified resolution details for a single break."""
    break_id: str = Field(description="ID of the break being resolved")
    corrected_value: str = Field(description="The exact corrected value or 'Manual review required'")
    reasoning: str = Field(description="Brief explanation of why this is the correct value")
    confidence: float = Field(ge=0, le=1, description="Confidence in the suggested value")


class BreaksResolutionResponse(BaseAgentResponse):
    """Simplified response from the breaks resolution agent."""
    resolutions: List[BreakResolution] = Field(description="List of break resolutions")
    
    # Summary statistics
    total_breaks_analyzed: int = Field(description="Total number of breaks analyzed")
    total_resolvable: int = Field(description="Number of breaks with suggested values")
    total_requiring_manual_review: int = Field(description="Number requiring manual review")


class BreaksResolutionAgent:
    """
    Agent for analyzing break root causes and suggesting resolutions.
    Takes breaks from the breaks identifier and provides actionable fixes.
    """
    
    def __init__(self):
        self.agent_name = "breaks_resolution_agent"
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for resolution analysis."""
        return """You are an expert financial reconciliation analyst. Your job is to provide EXACT CORRECTED VALUES for reconciliation breaks.

RULES:
1. For VALUE MISMATCHES: Calculate the exact correct number
2. For MISSING RECORDS: State "Manual review required"  
3. For TAX ISSUES: Use standard rates (US=15%, UK=20%, etc.) and calculate exact amounts
4. For QUANTITY ISSUES: Provide exact share count
5. For DATE ISSUES: Provide exact date in YYYY-MM-DD format
6. WHEN IN DOUBT: Assume the Custodian value is correct (Custodian systems are typically more reliable for settlement data)

EXAMPLES:
- Tax break: "450.00" (reasoning: "15% withholding tax on $3000 gross = $450")
- Quantity break: "2500" (reasoning: "Post 2:1 stock split, original 1250 shares becomes 2500")
- Missing record: "Manual review required" (reasoning: "Need to verify if dividend payment actually occurred")
- FX break: "1250.75" (reasoning: "USD amount $1000 * EUR/USD rate 1.25075")
- Uncertain case: "125.50" (reasoning: "Using Custodian value as it's more reliable for settlement data")

Always provide NUMBERS, DATES, or "Manual review required" - never vague descriptions.

Return JSON with corrected_value and reasoning for each break."""
    
    def _build_resolution_prompt(
        self,
        breaks: List[Dict[str, Any]],
        additional_context: Optional[str] = None,
        historical_patterns: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the prompt for resolution analysis."""
        
        prompt = f"""Please analyze these reconciliation breaks and provide ACTUAL CORRECTED VALUES for each break:

BREAKS TO ANALYZE ({len(breaks)} total):
```json
{json.dumps(breaks[:20], indent=2, default=str)}
```
{f'... and {len(breaks) - 20} more breaks' if len(breaks) > 20 else ''}
"""

        if additional_context:
            prompt += f"""

ADDITIONAL CONTEXT:
{additional_context}
"""

        if historical_patterns:
            prompt += f"""

HISTORICAL PATTERNS (from previous reconciliations):
```json
{json.dumps(historical_patterns, indent=2, default=str)}
```
"""

        prompt += """

For each break, provide:
1. The EXACT corrected value (number, text, or "Manual review required")
2. Brief reasoning explaining why this is the correct value

Return JSON in this simple format:

{
  "success": true,
  "resolutions": [
    {
      "break_id": "BRK_001",
      "corrected_value": "360.00",
      "reasoning": "US securities have 15% withholding tax rate, so correct tax = gross_amount * 0.15 = 2400 * 0.15 = 360.00",
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
  "total_requiring_manual_review": 1
}

Focus on providing exact corrected values with clear, brief reasoning."""
        
        return prompt
    
    async def analyze_and_resolve(
        self,
        breaks: List[Dict[str, Any]],
        additional_context: Optional[str] = None,
        historical_patterns: Optional[Dict[str, Any]] = None,
        timeout: float = 180.0,
        temperature: float = 0.1
    ) -> AgentResult:
        """
        Analyze breaks and provide resolution strategies.
        
        Args:
            breaks: List of breaks from breaks identifier agent
            additional_context: Additional context about the reconciliation
            historical_patterns: Historical patterns from previous reconciliations
            timeout: API timeout
            temperature: LLM temperature
            
        Returns:
            AgentResult with resolution strategies
        """
        
        logger.info(f"Analyzing {len(breaks)} breaks for resolution strategies")
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt()
            },
            {
                "role": "user",
                "content": self._build_resolution_prompt(
                    breaks, additional_context, historical_patterns
                )
            }
        ]
        
        # Execute agent
        executor = AgentExecutor()
        result = await executor.execute_single_agent(
            agent_name=self.agent_name,
            messages=messages,
            expected_response_type=BreaksResolutionResponse,
            timeout=timeout,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        return result
    
    def get_fixes_with_values(self, result: AgentResult) -> List[Dict[str, Any]]:
        """Get all fixes that have suggested values (not manual review)."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        resolutions = result.response.structured_data.get("resolutions", [])
        return [r for r in resolutions if r.get("corrected_value", "") != "Manual review required"]
    
    def get_manual_review_fixes(self, result: AgentResult) -> List[Dict[str, Any]]:
        """Get all fixes that require manual review."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        resolutions = result.response.structured_data.get("resolutions", [])
        return [r for r in resolutions if r.get("corrected_value", "") == "Manual review required"]
    
    def get_high_confidence_fixes(self, result: AgentResult, min_confidence: float = 0.8) -> List[Dict[str, Any]]:
        """Get fixes with high confidence scores."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        resolutions = result.response.structured_data.get("resolutions", [])
        return [r for r in resolutions if r.get("confidence", 0) >= min_confidence]
    
    def generate_fix_script(self, resolution: Dict[str, Any]) -> str:
        """
        Generate a Python script to apply the fix.
        
        Args:
            resolution: Single resolution from the agent
            
        Returns:
            Python script as string
        """
        
        corrected_value = resolution.get("corrected_value", "Manual review required")
        if corrected_value == "Manual review required":
            return ""
        
        script = f'''"""
Auto-generated fix script for Break: {resolution.get("break_id", "Unknown")}
Corrected Value: {corrected_value}
Reasoning: {resolution.get("reasoning", "No reasoning provided")}
"""

def apply_fix(data, break_id, field_name):
    """Apply the fix by updating the incorrect value."""
    
    corrected_value = "{corrected_value}"
    
    # Find the record with this break_id and update the field
    for record in data:
        if record.get("break_id") == break_id:
            # Update the field with the corrected value
            record[field_name] = corrected_value
            print(f"Updated {{field_name}} to {{corrected_value}} for break {{break_id}}")
            return True
    
    print(f"Could not find record for break {{break_id}}")
    return False

if __name__ == "__main__":
    # Example usage
    success = apply_fix(data, "{resolution.get("break_id", "Unknown")}", "field_name")
    print(f"Fix applied: {{success}}")
'''
        
        return script
    
    def validate_resolution(self, result: AgentResult) -> bool:
        """Validate that the resolution result is usable."""
        
        if result.status != AgentStatus.COMPLETED:
            logger.error(f"Agent failed: {result.error}")
            return False
            
        if not result.response or not result.response.structured_data:
            logger.error("No structured data in response")
            return False
            
        data = result.response.structured_data
        if "resolutions" not in data:
            logger.error("Missing resolutions in response")
            return False
            
        return True


# Example usage
async def example_usage():
    """Example of using the breaks resolution agent."""
    
    # Sample breaks from breaks identifier (with composite keys)
    sample_breaks = [
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
            "impact_assessment": "Net settlement amount will be incorrect",
            "confidence": 0.95
        },
        {
            "break_id": "BRK_002",
            "description": "Missing record in Custody system",
            "classification": "Missing Record",
            "severity": "Critical",
            "composite_key": "EVT002|US5949181045|2588173|ACC12345",
            "composite_key_components": {
                "coac_event_key": "EVT002",
                "isin": "US5949181045",
                "sedol": "2588173",
                "bank_account": "ACC12345"
            },
            "affected_field": None,
            "nbim_value": "Full record exists",
            "custody_value": "No record found",
            "difference": "Complete record missing",
            "impact_assessment": "Dividend payment not tracked in Custody",
            "confidence": 1.0
        },
        {
            "break_id": "BRK_003",
            "description": "Quantity mismatch for position",
            "classification": "Value Discrepancy",
            "severity": "High",
            "composite_key": "EVT003|US0378331005|2046251|ACC67890",
            "composite_key_components": {
                "coac_event_key": "EVT003",
                "isin": "US0378331005",
                "sedol": "2046251",
                "bank_account": "ACC67890"
            },
            "affected_field": "quantity",
            "nbim_value": 5000,
            "custody_value": 5500,
            "difference": -500,
            "impact_assessment": "Incorrect dividend calculation",
            "confidence": 0.9
        }
    ]
    
    # Additional context
    context = """
    This is Q1 2024 dividend reconciliation.
    Standard US withholding tax rate is 15% per treaty.
    NBIM is the book of record.
    Custody data comes from State Street.
    Settlement is typically T+2 for US securities.
    """
    
    # Historical patterns (if available)
    historical_patterns = {
        "common_issues": [
            "Tax rate mismatches for US securities",
            "Timing differences for ex-date near month-end",
            "Quantity discrepancies for stock splits"
        ],
        "previous_fixes": {
            "tax_rate_issues": "Usually Custody system needs update",
            "timing_differences": "Usually resolve within 2 days"
        }
    }
    
    # Create and run the agent
    agent = BreaksResolutionAgent()
    
    print("🔧 Analyzing breaks for resolution strategies...")
    result = await agent.analyze_and_resolve(
        breaks=sample_breaks,
        additional_context=context,
        historical_patterns=historical_patterns
    )
    
    # Process results
    if agent.validate_resolution(result):
        data = result.response.structured_data
        
        print("✅ Resolution analysis complete!")
        print(f"\n📊 Summary:")
        print(f"  Total breaks analyzed: {data.get('total_breaks_analyzed', 0)}")
        print(f"  Resolvable breaks: {data.get('total_resolvable', 0)}")
        print(f"  Requiring manual review: {data.get('total_requiring_manual_review', 0)}")
        
        # Root cause summary
        print(f"\n🔍 Root Causes Identified:")
        for cause, count in data.get('root_cause_summary', {}).items():
            print(f"  - {cause}: {count}")
        
        # Fix type summary
        print(f"\n🛠️ Fix Types:")
        for fix_type, count in data.get('fix_type_summary', {}).items():
            print(f"  - {fix_type}: {count}")
        
        # Automation potential
        automation = data.get('automation_potential', {})
        print(f"\n🤖 Automation Potential:")
        print(f"  - Fully automatable: {automation.get('fully_automatable', 0)}")
        print(f"  - Partially automatable: {automation.get('partially_automatable', 0)}")
        print(f"  - Manual only: {automation.get('manual_only', 0)}")
        
        # Show sample resolutions
        print(f"\n📋 Sample Resolutions:")
        for resolution in data.get('resolutions', [])[:2]:
            print(f"\n  Break ID: {resolution.get('break_id')}")
            print(f"  🔑 Composite Key: {resolution.get('composite_key')}")
            print(f"  Root Cause: {resolution.get('root_cause')}")
            print(f"  Fix Type: {resolution.get('fix_type')}")
            print(f"  Complexity: {resolution.get('fix_complexity')}")
            print(f"  Can Automate: {'Yes' if resolution.get('can_be_automated') else 'No'}")
            print(f"  Priority: {resolution.get('priority')}")
            print(f"  Steps:")
            for step in resolution.get('resolution_steps', [])[:2]:
                print(f"    {step.get('step_number')}. {step.get('action')} ({step.get('target_system')})")
        
        # Systemic issues
        systemic_issues = data.get('systemic_issues_identified', [])
        if systemic_issues:
            print(f"\n⚠️ Systemic Issues Found:")
            for issue in systemic_issues:
                print(f"  - {issue}")
        
        # Process improvements
        improvements = data.get('process_improvements', [])
        if improvements:
            print(f"\n💡 Recommended Process Improvements:")
            for improvement in improvements:
                print(f"  - {improvement}")
        
        # Get automatable fixes
        automatable = agent.get_automatable_fixes(result)
        if automatable:
            print(f"\n🤖 Automatable Fixes: {len(automatable)}")
            
            # Generate sample fix script
            if automatable:
                print(f"\n📝 Sample Fix Script for {automatable[0].get('break_id')}:")
                script = agent.generate_fix_script(automatable[0])
                if script:
                    print("```python")
                    print(script[:500] + "..." if len(script) > 500 else script)
                    print("```")
        
    else:
        print(f"❌ Resolution analysis failed: {result.error if result.error else 'Unknown error'}")


if __name__ == "__main__":
    asyncio.run(example_usage())