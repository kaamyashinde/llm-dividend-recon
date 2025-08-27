"""
JIRA Issue Agent - Converts reconciliation breaks into JIRA-importable format.

This agent takes identified breaks and resolution information and formats them
into proper JIRA tickets that can be imported via CSV or created via API.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import asyncio
import json
from datetime import datetime
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


class JiraIssuePriority(str, Enum):
    """JIRA issue priority levels."""
    HIGHEST = "Highest"
    HIGH = "High"  
    MEDIUM = "Medium"
    LOW = "Low"
    LOWEST = "Lowest"


class JiraIssueType(str, Enum):
    """JIRA issue types."""
    TASK = "Task"
    BUG = "Bug"
    STORY = "Story"
    IMPROVEMENT = "Improvement"
    SUB_TASK = "Sub-task"


class JiraIssue(BaseModel):
    """A single JIRA issue formatted for import."""
    summary: str = Field(description="Brief summary of the issue")
    description: str = Field(description="Detailed description with break information")
    issue_type: JiraIssueType = Field(description="Type of JIRA issue")
    priority: JiraIssuePriority = Field(description="Priority level")
    
    # Optional fields that may be available in JIRA
    component: Optional[str] = Field(None, description="Component/area (e.g., 'Dividend Reconciliation')")
    labels: List[str] = Field(default_factory=list, description="Labels for categorization")
    assignee: Optional[str] = Field(None, description="Assignee username or email")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")
    
    # Custom fields for reconciliation context
    break_id: str = Field(description="Original break ID from reconciliation")
    break_classification: str = Field(description="Classification of the break")
    affected_field: Optional[str] = Field(None, description="Field where break was found")
    nbim_value: Optional[str] = Field(None, description="NBIM value as string")
    custody_value: Optional[str] = Field(None, description="Custody value as string")
    financial_impact: Optional[str] = Field(None, description="Financial impact description")
    
    # For CSV import format
    def to_csv_row(self) -> Dict[str, str]:
        """Convert to CSV row format for JIRA import."""
        return {
            "Summary": self.summary,
            "Description": self.description,
            "Issue Type": self.issue_type.value,
            "Priority": self.priority.value,
            "Component": self.component or "",
            "Labels": ";".join(self.labels) if self.labels else "",
            "Assignee": self.assignee or "",
            "Due Date": self.due_date or "",
            # Custom fields - these would need to be configured in JIRA
            "Break ID": self.break_id,
            "Break Classification": self.break_classification,
            "Affected Field": self.affected_field or "",
            "NBIM Value": self.nbim_value or "",
            "Custody Value": self.custody_value or "",
            "Financial Impact": self.financial_impact or ""
        }


class JiraIssueResponse(BaseAgentResponse):
    """Response from the JIRA issue agent."""
    jira_issues: List[JiraIssue] = Field(description="List of JIRA issues to create")
    
    # Summary statistics
    total_issues_created: int = Field(description="Total number of JIRA issues created")
    issues_by_priority: Dict[str, int] = Field(description="Count of issues by priority")
    issues_by_type: Dict[str, int] = Field(description="Count of issues by type")
    
    # Export formats
    csv_headers: List[str] = Field(description="Headers for CSV import")
    csv_data: List[Dict[str, str]] = Field(description="Data rows for CSV import")
    
    # Metadata
    created_timestamp: str = Field(description="Timestamp when issues were created")
    total_breaks_processed: int = Field(description="Number of breaks processed")
    estimated_resolution_time: str = Field(description="Estimated time to resolve all issues")


class JiraIssueAgent:
    """
    Agent for creating JIRA-compatible issues from reconciliation breaks.
    Converts breaks and resolution data into proper JIRA import format.
    """
    
    def __init__(self):
        self.agent_name = "jira_issue_agent"
        self.default_component = "Dividend Reconciliation"
        self.default_labels = ["reconciliation", "dividend", "automated"]
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for JIRA issue creation."""
        return """You are an expert JIRA issue creator specializing in financial reconciliation problems.

Your task is to convert reconciliation breaks into proper JIRA tickets that can be imported and worked on by a reconciliation team.

JIRA ISSUE GUIDELINES:
1. SUMMARY: Clear, actionable title (max 100 chars) - start with action verb
2. DESCRIPTION: Detailed explanation with context, impact, and steps
3. PRIORITY: Map break severity to JIRA priority (Critical->Highest, High->High, etc.)
4. ISSUE TYPE: Choose appropriate type (Task for processes, Bug for system errors, etc.)
5. LABELS: Add relevant categorization labels
6. COMPONENT: Use "Dividend Reconciliation" unless break suggests otherwise

PRIORITY MAPPING:
- Critical breaks → Highest priority
- High breaks → High priority  
- Medium breaks → Medium priority
- Low breaks → Low priority
- Unknown → Medium priority

ISSUE TYPE MAPPING:
- Missing records → Bug (data not where expected)
- Value mismatches → Task (investigation needed)
- System errors → Bug (technical problem)
- Process issues → Improvement (process enhancement)
- Manual review needed → Task (human action required)

DESCRIPTION FORMAT:
```
## Break Details
- Break ID: {break_id}
- Classification: {classification}
- Field: {affected_field}
- NBIM Value: {nbim_value}
- Custody Value: {custody_value}

## Impact Assessment
{impact_description}

## Required Action
{suggested_resolution_or_investigation_steps}

## Context
{additional_context_about_the_break}
```

SUMMARY EXAMPLES:
- "Investigate missing dividend record for AAPL event EVT001"
- "Fix withholding tax calculation for US securities" 
- "Resolve quantity mismatch in MSFT position ACC123"
- "Update FX rate for EUR dividend payment"

Return structured JSON matching the JiraIssueResponse schema."""
    
    def _build_jira_creation_prompt(
        self,
        breaks: List[Dict[str, Any]],
        resolution_data: Optional[Dict[str, Any]] = None,
        jira_config: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """Build the prompt for creating JIRA issues."""
        
        prompt = f"""Please convert these reconciliation breaks into JIRA-compatible issues:

RECONCILIATION BREAKS ({len(breaks)} total):
```json
{json.dumps(breaks, indent=2, default=str)}
```
"""

        if resolution_data:
            prompt += f"""
RESOLUTION ANALYSIS DATA:
```json
{json.dumps(resolution_data, indent=2, default=str)}
```
Use resolution suggestions to enhance JIRA issue descriptions and determine priorities.
"""

        if jira_config:
            prompt += f"""
JIRA PROJECT CONFIGURATION:
```json
{json.dumps(jira_config, indent=2, default=str)}
```
Follow any specific configuration requirements.
"""

        if additional_context:
            prompt += f"""
ADDITIONAL CONTEXT:
{additional_context}
"""

        prompt += f"""
REQUIREMENTS:
1. Create ONE JIRA issue for each break (1:1 mapping)
2. Use break_id to link JIRA issue back to original break
3. Map break severity to appropriate JIRA priority
4. Choose suitable issue type based on break classification
5. Include all relevant break details in description
6. Add appropriate labels for filtering and reporting
7. Set component to "{self.default_component}" unless break suggests otherwise
8. Provide actionable summary titles that describe what needs to be done

OUTPUT FORMAT: Return JSON matching this structure:
{{
  "success": true,
  "jira_issues": [
    {{
      "summary": "Fix withholding tax calculation for Apple dividend",
      "description": "## Break Details\\n- Break ID: BRK_001\\n- Classification: Value Discrepancy\\n...",
      "issue_type": "Task",
      "priority": "High",
      "component": "Dividend Reconciliation",
      "labels": ["reconciliation", "dividend", "tax", "value-mismatch"],
      "assignee": null,
      "due_date": null,
      "break_id": "BRK_001",
      "break_classification": "Value Discrepancy",
      "affected_field": "withholding_tax",
      "nbim_value": "360.00",
      "custody_value": "300.00",
      "financial_impact": "$60 difference affects net settlement"
    }}
  ],
  "total_issues_created": 1,
  "issues_by_priority": {{"High": 1}},
  "issues_by_type": {{"Task": 1}},
  "csv_headers": ["Summary", "Description", "Issue Type", "Priority", "Component", "Labels", "Break ID"],
  "csv_data": [
    {{"Summary": "Fix withholding tax calculation for Apple dividend", "Description": "...", "Issue Type": "Task"}}
  ],
  "created_timestamp": "{datetime.now().isoformat()}",
  "total_breaks_processed": {len(breaks)},
  "estimated_resolution_time": "2-4 hours per issue based on complexity"
}}

Focus on creating actionable, well-structured JIRA issues that a reconciliation team can immediately work with."""
        
        return prompt
    
    async def create_jira_issues(
        self,
        breaks: List[Dict[str, Any]],
        resolution_data: Optional[Dict[str, Any]] = None,
        jira_config: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None,
        timeout: float = 120.0,
        temperature: float = 0.2
    ) -> AgentResult:
        """
        Create JIRA issues from reconciliation breaks.
        
        Args:
            breaks: List of breaks from breaks identifier agent
            resolution_data: Optional resolution analysis data
            jira_config: Optional JIRA project configuration
            additional_context: Additional context for issue creation
            timeout: API timeout
            temperature: LLM temperature (lower = more consistent)
            
        Returns:
            AgentResult with JIRA issues data
        """
        
        logger.info(f"Creating JIRA issues for {len(breaks)} reconciliation breaks")
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt()
            },
            {
                "role": "user",
                "content": self._build_jira_creation_prompt(
                    breaks, resolution_data, jira_config, additional_context
                )
            }
        ]
        
        # Execute agent
        executor = AgentExecutor()
        result = await executor.execute_single_agent(
            agent_name=self.agent_name,
            messages=messages,
            expected_response_type=JiraIssueResponse,
            timeout=timeout,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        return result
    
    def format_for_csv_import(self, result: AgentResult) -> Optional[str]:
        """
        Format JIRA issues as CSV for bulk import.
        
        Args:
            result: Agent result containing JIRA issues
            
        Returns:
            CSV string ready for JIRA import, or None if failed
        """
        
        if not result.response or not result.response.structured_data:
            logger.error("No JIRA issues data to format")
            return None
        
        data = result.response.structured_data
        jira_issues = data.get("jira_issues", [])
        
        if not jira_issues:
            logger.warning("No JIRA issues found to format")
            return None
        
        try:
            import csv
            from io import StringIO
            
            output = StringIO()
            
            # Get CSV headers and data from the response
            csv_headers = data.get("csv_headers", [])
            csv_data = data.get("csv_data", [])
            
            if not csv_headers or not csv_data:
                # Fallback: create CSV from jira_issues directly
                if jira_issues:
                    first_issue = jira_issues[0]
                    csv_headers = list(first_issue.keys()) if isinstance(first_issue, dict) else [
                        "Summary", "Description", "Issue Type", "Priority", "Component", 
                        "Labels", "Break ID", "Break Classification"
                    ]
                    
                    csv_data = []
                    for issue in jira_issues:
                        if isinstance(issue, dict):
                            row = {header: str(issue.get(header.lower().replace(" ", "_"), "")) for header in csv_headers}
                        else:
                            # If it's a Pydantic model, convert to dict
                            issue_dict = issue if isinstance(issue, dict) else getattr(issue, '__dict__', {})
                            row = {}
                            for header in csv_headers:
                                key = header.lower().replace(" ", "_")
                                value = issue_dict.get(key, "")
                                if key == "labels" and isinstance(value, list):
                                    value = ";".join(value)
                                row[header] = str(value)
                        csv_data.append(row)
            
            # Write CSV
            writer = csv.DictWriter(output, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(csv_data)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error formatting JIRA issues as CSV: {e}")
            return None
    
    def format_for_json_import(self, result: AgentResult) -> Optional[str]:
        """
        Format JIRA issues as JSON for API import.
        
        Args:
            result: Agent result containing JIRA issues
            
        Returns:
            JSON string ready for JIRA API, or None if failed
        """
        
        if not result.response or not result.response.structured_data:
            return None
        
        data = result.response.structured_data
        jira_issues = data.get("jira_issues", [])
        
        if not jira_issues:
            return None
        
        try:
            # Format for JIRA API structure
            jira_api_format = {
                "issues": [
                    {
                        "fields": {
                            "summary": issue.get("summary", ""),
                            "description": issue.get("description", ""),
                            "issuetype": {"name": issue.get("issue_type", "Task")},
                            "priority": {"name": issue.get("priority", "Medium")},
                            "components": [{"name": issue.get("component", self.default_component)}] if issue.get("component") else [],
                            "labels": issue.get("labels", self.default_labels),
                            # Custom fields would go here with proper field IDs
                            "customfield_break_id": issue.get("break_id", ""),
                            "customfield_break_classification": issue.get("break_classification", ""),
                            "customfield_affected_field": issue.get("affected_field", ""),
                            "customfield_nbim_value": issue.get("nbim_value", ""),
                            "customfield_custody_value": issue.get("custody_value", ""),
                            "customfield_financial_impact": issue.get("financial_impact", "")
                        }
                    }
                    for issue in jira_issues
                ]
            }
            
            return json.dumps(jira_api_format, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error formatting JIRA issues as JSON: {e}")
            return None
    
    def get_issue_summary(self, result: AgentResult) -> Dict[str, Any]:
        """
        Get a summary of created JIRA issues.
        
        Args:
            result: Agent result containing JIRA issues
            
        Returns:
            Summary dictionary with counts and statistics
        """
        
        if not result.response or not result.response.structured_data:
            return {"error": "No JIRA issues data available"}
        
        data = result.response.structured_data
        
        return {
            "total_issues": data.get("total_issues_created", 0),
            "by_priority": data.get("issues_by_priority", {}),
            "by_type": data.get("issues_by_type", {}),
            "total_breaks_processed": data.get("total_breaks_processed", 0),
            "estimated_resolution_time": data.get("estimated_resolution_time", "Unknown"),
            "created_timestamp": data.get("created_timestamp", ""),
            "status": "success" if result.status == AgentStatus.COMPLETED else "failed"
        }
    
    def validate_jira_issues(self, result: AgentResult) -> bool:
        """Validate that JIRA issues were created successfully."""
        
        if result.status != AgentStatus.COMPLETED:
            logger.error(f"JIRA agent failed: {result.error}")
            return False
            
        if not result.response or not result.response.structured_data:
            logger.error("No structured data in JIRA response")
            return False
            
        data = result.response.structured_data
        if "jira_issues" not in data:
            logger.error("Missing jira_issues in response")
            return False
            
        jira_issues = data.get("jira_issues", [])
        if not jira_issues:
            logger.warning("No JIRA issues created (empty list)")
            return False
        
        # Validate individual issues have required fields
        required_fields = ["summary", "description", "issue_type", "priority", "break_id"]
        for i, issue in enumerate(jira_issues):
            for field in required_fields:
                if not issue.get(field):
                    logger.error(f"JIRA issue {i} missing required field: {field}")
                    return False
        
        logger.info(f"Successfully validated {len(jira_issues)} JIRA issues")
        return True


# Example usage
async def example_usage():
    """Example of using the JIRA issue agent."""
    
    # Sample breaks from breaks identifier (matching the format from breaks_identifier_agent.py)
    sample_breaks = [
        {
            "break_id": "BRK_001",
            "description": "Tax amount mismatch between NBIM and Custody systems",
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
            "impact_assessment": "Net settlement amount will be incorrect by $60",
            "confidence": 0.95
        },
        {
            "break_id": "BRK_002", 
            "description": "Missing dividend record in Custody system",
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
            "impact_assessment": "$3,750 dividend payment not tracked in Custody",
            "confidence": 1.0
        },
        {
            "break_id": "BRK_003",
            "description": "Quantity mismatch for Apple position",
            "classification": "Value Discrepancy", 
            "severity": "Medium",
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
            "impact_assessment": "Dividend calculation incorrect by $120",
            "confidence": 0.9
        }
    ]
    
    # Sample resolution data (from breaks_resolution_agent.py format)
    resolution_data = {
        "resolutions": [
            {
                "break_id": "BRK_001",
                "corrected_value": "360.00",
                "reasoning": "US securities have 15% withholding tax rate. Correct tax = $2400 * 0.15 = $360",
                "confidence": 0.95
            },
            {
                "break_id": "BRK_002",
                "corrected_value": "Manual review required",
                "reasoning": "Missing record requires investigation of source systems to determine if dividend was actually paid",
                "confidence": 0.0
            },
            {
                "break_id": "BRK_003",
                "corrected_value": "5500",
                "reasoning": "Custody quantity appears correct based on settlement confirmations",
                "confidence": 0.8
            }
        ],
        "total_breaks_analyzed": 3,
        "total_resolvable": 2,
        "total_requiring_manual_review": 1
    }
    
    # JIRA project configuration
    jira_config = {
        "project_key": "RECON",
        "default_assignee": "recon-team@company.com",
        "components": ["Dividend Reconciliation", "Data Quality"],
        "labels": ["Q1-2024", "automated", "reconciliation"],
        "due_date_offset_days": 7  # Issues due in 7 days
    }
    
    # Additional context
    additional_context = """
    This is Q1 2024 dividend reconciliation for Norwegian Bank Investment Management.
    All issues should be resolved before month-end closing.
    High and Critical priority issues need immediate attention.
    Contact reconciliation team for clarification on manual review items.
    """
    
    # Create and run the agent
    agent = JiraIssueAgent()
    
    print("🎫 Creating JIRA issues from reconciliation breaks...")
    result = await agent.create_jira_issues(
        breaks=sample_breaks,
        resolution_data=resolution_data,
        jira_config=jira_config,
        additional_context=additional_context
    )
    
    # Process results
    if agent.validate_jira_issues(result):
        data = result.response.structured_data
        
        print("✅ JIRA issues created successfully!")
        
        # Show summary
        summary = agent.get_issue_summary(result)
        print(f"\n📊 Summary:")
        print(f"  Total JIRA issues created: {summary['total_issues']}")
        print(f"  Breaks processed: {summary['total_breaks_processed']}")
        print(f"  Estimated resolution time: {summary['estimated_resolution_time']}")
        
        # Show breakdown by priority and type
        print(f"\n🎯 By Priority:")
        for priority, count in summary['by_priority'].items():
            print(f"  - {priority}: {count}")
        
        print(f"\n📋 By Issue Type:")
        for issue_type, count in summary['by_type'].items():
            print(f"  - {issue_type}: {count}")
        
        # Show sample JIRA issues
        print(f"\n🎫 Sample JIRA Issues:")
        for issue in data.get('jira_issues', [])[:2]:
            print(f"\n  📌 {issue.get('summary', 'N/A')}")
            print(f"     Type: {issue.get('issue_type', 'N/A')} | Priority: {issue.get('priority', 'N/A')}")
            print(f"     Break ID: {issue.get('break_id', 'N/A')}")
            print(f"     Labels: {', '.join(issue.get('labels', []))}")
            # Show first 100 chars of description
            desc = issue.get('description', '')
            if len(desc) > 100:
                desc = desc[:100] + "..."
            print(f"     Description: {desc}")
        
        # Generate export formats
        print(f"\n📥 Export Options:")
        
        # CSV format
        csv_data = agent.format_for_csv_import(result)
        if csv_data:
            print(f"  ✅ CSV format ready ({len(csv_data)} characters)")
            print(f"     First few lines:")
            lines = csv_data.split('\n')
            for line in lines[:3]:
                print(f"     {line}")
            if len(lines) > 3:
                print(f"     ... and {len(lines) - 3} more lines")
        
        # JSON format  
        json_data = agent.format_for_json_import(result)
        if json_data:
            print(f"  ✅ JSON API format ready ({len(json_data)} characters)")
        
        print(f"\n💡 Next Steps:")
        print(f"  1. Download the CSV file and import into JIRA")
        print(f"  2. Configure custom fields in JIRA for break tracking")
        print(f"  3. Assign issues to appropriate team members")
        print(f"  4. Set up JIRA automation rules for status updates")
        
    else:
        print(f"❌ JIRA issue creation failed: {result.error if result.error else 'Unknown error'}")


if __name__ == "__main__":
    asyncio.run(example_usage())