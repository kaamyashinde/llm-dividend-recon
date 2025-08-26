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


class ResolutionStep(BaseModel):
    """A single step in the resolution process."""
    step_number: int = Field(description="Order of execution for this step")
    action: str = Field(description="Specific action to take")
    target_system: str = Field(description="System where action should be applied (NBIM/Custody/Both)")
    details: str = Field(description="Detailed instructions for this step")
    automated_possible: bool = Field(description="Whether this step can be automated")


class BreakResolution(BaseModel):
    """Resolution details for a single break."""
    break_id: str = Field(description="ID of the break being resolved")
    composite_key: str = Field(description="Composite key of the affected record")
    
    # Root cause analysis
    root_cause: str = Field(description="Identified root cause of the break")
    root_cause_category: str = Field(description="Category of root cause")
    root_cause_confidence: float = Field(ge=0, le=1, description="Confidence in root cause identification")
    
    # Supporting evidence
    evidence: List[str] = Field(description="Evidence supporting the root cause analysis")
    patterns_detected: List[str] = Field(default=[], description="Patterns that helped identify the cause")
    
    # Suggested fix
    fix_type: str = Field(description="Type of fix recommended")
    fix_description: str = Field(description="Detailed description of the suggested fix")
    fix_complexity: str = Field(description="Complexity level (Simple/Medium/Complex)")
    
    # Resolution steps
    resolution_steps: List[ResolutionStep] = Field(description="Step-by-step resolution process")
    
    # Expected outcome
    expected_result: str = Field(description="Expected result after applying the fix")
    success_criteria: List[str] = Field(description="Criteria to verify successful resolution")
    
    # Risk assessment
    risk_level: str = Field(description="Risk level of applying this fix (Low/Medium/High)")
    risk_factors: List[str] = Field(default=[], description="Potential risks if fix is applied")
    
    # Automation potential
    can_be_automated: bool = Field(description="Whether this fix can be fully automated")
    automation_confidence: float = Field(ge=0, le=1, description="Confidence in automation")
    manual_validation_required: bool = Field(description="Whether manual validation is needed")
    
    # Alternative solutions
    alternative_fixes: List[str] = Field(default=[], description="Alternative ways to fix this break")
    
    # Priority and urgency
    priority: str = Field(description="Priority level for this fix (Critical/High/Medium/Low)")
    estimated_time_to_fix: str = Field(description="Estimated time to implement the fix")


class BreaksResolutionResponse(BaseAgentResponse):
    """Response from the breaks resolution agent."""
    resolutions: List[BreakResolution] = Field(description="List of break resolutions")
    
    # Summary statistics
    total_breaks_analyzed: int = Field(description="Total number of breaks analyzed")
    total_resolvable: int = Field(description="Number of breaks that can be resolved")
    total_requiring_manual_review: int = Field(description="Number requiring manual review")
    
    # Root cause summary
    root_cause_summary: Dict[str, int] = Field(description="Count of breaks by root cause category")
    fix_type_summary: Dict[str, int] = Field(description="Count of breaks by fix type")
    
    # Automation potential
    automation_potential: Dict[str, Any] = Field(description="Summary of automation possibilities")
    
    # Overall recommendations
    overall_recommendations: List[str] = Field(description="High-level recommendations")
    systemic_issues_identified: List[str] = Field(description="Systemic issues found across breaks")
    
    # Process improvements
    process_improvements: List[str] = Field(description="Suggested process improvements")
    
    # Metadata
    analysis_timestamp: str = Field(description="When the analysis was performed")
    confidence_score: float = Field(ge=0, le=1, description="Overall confidence in resolutions")


class BreaksResolutionAgent:
    """
    Agent for analyzing break root causes and suggesting resolutions.
    Takes breaks from the breaks identifier and provides actionable fixes.
    """
    
    def __init__(self):
        self.agent_name = "breaks_resolution_agent"
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for resolution analysis."""
        return """You are an expert financial operations analyst specializing in break resolution and reconciliation fixes.

Your task is to analyze reconciliation breaks and provide detailed resolution strategies.

For each break, you must:

1. ROOT CAUSE ANALYSIS:
   - Identify the most likely root cause based on the break details
   - Categorize the root cause (e.g., timing difference, data entry error, system sync issue, calculation error)
   - Provide evidence supporting your conclusion
   - Identify any patterns that indicate systemic issues

2. RESOLUTION STRATEGY:
   - Propose a specific, actionable fix
   - Break down the fix into clear, sequential steps
   - Identify which system needs updating (NBIM, Custody, or both)
   - Assess if the fix can be automated

3. RISK ASSESSMENT:
   - Evaluate the risk of applying the fix
   - Identify potential side effects
   - Suggest validation steps

4. COMMON ROOT CAUSES TO CONSIDER:
   - Timing differences (T+1, T+2 settlement)
   - Tax rate discrepancies (withholding tax rates)
   - FX rate differences (spot vs forward rates)
   - Corporate action processing delays
   - Manual data entry errors
   - System synchronization issues
   - Calculation methodology differences
   - Rounding differences
   - Missing reference data
   - Duplicate entries
   - Partial settlements
   - Fee calculations

5. FIX CATEGORIES:
   - Data corrections (update incorrect values)
   - Recalculations (recompute using correct formula)
   - Date alignments (adjust for timing differences)
   - Reference data updates (update static data)
   - System synchronization (align data between systems)
   - Manual review required (complex cases)

Be specific and practical in your recommendations. Focus on fixes that can actually be implemented.

Return structured JSON following the BreaksResolutionResponse schema."""
    
    def _build_resolution_prompt(
        self,
        breaks: List[Dict[str, Any]],
        additional_context: Optional[str] = None,
        historical_patterns: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the prompt for resolution analysis."""
        
        prompt = f"""Please analyze these reconciliation breaks and provide detailed resolution strategies:

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

For each break, provide a comprehensive resolution strategy including:
1. Root cause identification with evidence
2. Specific fix with step-by-step instructions
3. Risk assessment and automation potential
4. Alternative solutions if applicable

Focus on practical, implementable solutions. Consider:
- Whether the fix can be automated
- The impact on downstream processes
- Validation steps needed
- Time and effort required

Return your analysis as JSON matching this structure:

{
  "success": true,
  "resolutions": [
    {
      "break_id": "BRK_001",
      "composite_key": "EVT001|US123|SEDOL123|ACC001",
      "root_cause": "Withholding tax rate mismatch - NBIM using 15% while Custody using 10%",
      "root_cause_category": "Tax Rate Discrepancy",
      "root_cause_confidence": 0.95,
      "evidence": [
        "NBIM tax amount is exactly 15% of gross",
        "Custody tax amount is exactly 10% of gross",
        "This security is US-listed with standard 15% treaty rate"
      ],
      "patterns_detected": ["Consistent 5% difference across all US securities"],
      "fix_type": "tax_rate_update",
      "fix_description": "Update Custody system to apply correct 15% US withholding tax rate",
      "fix_complexity": "Simple",
      "resolution_steps": [
        {
          "step_number": 1,
          "action": "Update tax rate",
          "target_system": "Custody",
          "details": "Change withholding tax rate from 10% to 15% for US securities",
          "automated_possible": true
        },
        {
          "step_number": 2,
          "action": "Recalculate tax amount",
          "target_system": "Custody",
          "details": "Recalculate: tax = gross_amount * 0.15",
          "automated_possible": true
        },
        {
          "step_number": 3,
          "action": "Update net amount",
          "target_system": "Custody",
          "details": "Recalculate: net = gross_amount - tax",
          "automated_possible": true
        }
      ],
      "expected_result": "Tax and net amounts will match between systems",
      "success_criteria": [
        "Custody tax amount equals NBIM tax amount",
        "Custody net amount equals NBIM net amount"
      ],
      "risk_level": "Low",
      "risk_factors": ["May affect other US securities if applied broadly"],
      "can_be_automated": true,
      "automation_confidence": 0.9,
      "manual_validation_required": false,
      "alternative_fixes": [
        "Manual adjustment of this specific record",
        "Investigate if NBIM rate is incorrect instead"
      ],
      "priority": "High",
      "estimated_time_to_fix": "5 minutes"
    }
  ],
  "total_breaks_analyzed": 10,
  "total_resolvable": 8,
  "total_requiring_manual_review": 2,
  "root_cause_summary": {
    "Tax Rate Discrepancy": 4,
    "Timing Difference": 3,
    "Data Entry Error": 2,
    "Missing Record": 1
  },
  "fix_type_summary": {
    "tax_rate_update": 4,
    "date_alignment": 3,
    "data_correction": 2,
    "manual_review_required": 1
  },
  "automation_potential": {
    "fully_automatable": 6,
    "partially_automatable": 2,
    "manual_only": 2,
    "automation_percentage": 60
  },
  "overall_recommendations": [
    "Implement automated tax rate validation",
    "Standardize date formats between systems",
    "Add reconciliation checkpoints"
  ],
  "systemic_issues_identified": [
    "Consistent tax rate mismatch for US securities",
    "Date format inconsistencies"
  ],
  "process_improvements": [
    "Add pre-reconciliation data validation",
    "Implement real-time sync between systems"
  ],
  "analysis_timestamp": "2024-01-15T10:30:00Z",
  "confidence_score": 0.85
}

Be thorough in your analysis and provide actionable, specific fixes."""
        
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
    
    def get_automatable_fixes(self, result: AgentResult) -> List[Dict[str, Any]]:
        """Get all fixes that can be automated."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        resolutions = result.response.structured_data.get("resolutions", [])
        return [r for r in resolutions if r.get("can_be_automated", False)]
    
    def get_fixes_by_priority(self, result: AgentResult, priority: str) -> List[Dict[str, Any]]:
        """Get all fixes of a specific priority."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        resolutions = result.response.structured_data.get("resolutions", [])
        return [r for r in resolutions if r.get("priority", "").lower() == priority.lower()]
    
    def get_fixes_by_type(self, result: AgentResult, fix_type: str) -> List[Dict[str, Any]]:
        """Get all fixes of a specific type."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        resolutions = result.response.structured_data.get("resolutions", [])
        return [r for r in resolutions if r.get("fix_type") == fix_type]
    
    def get_systemic_issues(self, result: AgentResult) -> List[str]:
        """Extract systemic issues identified across multiple breaks."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        return result.response.structured_data.get("systemic_issues_identified", [])
    
    def generate_fix_script(self, resolution: Dict[str, Any]) -> str:
        """
        Generate a Python script to apply the fix (if automatable).
        
        Args:
            resolution: Single resolution from the agent
            
        Returns:
            Python script as string, or empty string if not automatable
        """
        
        if not resolution.get("can_be_automated", False):
            return ""
        
        script = f'''"""
Auto-generated fix script for Break: {resolution.get("break_id", "Unknown")}
Composite Key: {resolution.get("composite_key", "Unknown")}
Fix Type: {resolution.get("fix_type", "Unknown")}
Generated at: {resolution.get("analysis_timestamp", "Unknown")}
"""

def apply_fix(nbim_data, custody_data, composite_key):
    """Apply the automated fix for this break."""
    
    # Fix: {resolution.get("fix_description", "No description")}
    
'''
        
        # Add steps from resolution
        for step in resolution.get("resolution_steps", []):
            if step.get("automated_possible", False):
                script += f'''
    # Step {step.get("step_number")}: {step.get("action")}
    # Target: {step.get("target_system")}
    # Details: {step.get("details")}
    # TODO: Implement this step
    
'''
        
        script += '''
    return True  # Return success status

if __name__ == "__main__":
    # Example usage
    success = apply_fix(nbim_data, custody_data, composite_key)
    print(f"Fix applied: {success}")
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