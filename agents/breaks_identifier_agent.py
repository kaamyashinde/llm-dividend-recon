"""
Breaks Identifier Agent - Pure LLM-driven break identification and classification.

This agent uses an LLM to dynamically identify and classify breaks without hardcoded categories.
The LLM determines what constitutes a break and how to classify it based on the data context.
"""

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import asyncio
import json

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


# Minimal response models - let the LLM define the structure
class BreakIdentification(BaseModel):
    """A single identified break/discrepancy."""
    break_id: str = Field(description="Unique identifier for this break")
    description: str = Field(description="Clear description of the discrepancy")
    
    # Let LLM determine classification
    classification: str = Field(description="Type/category of break as determined by analysis")
    severity: str = Field(description="Severity level as determined by analysis")
    
    # Composite key fields for unique identification
    composite_key: str = Field(description="Composite key built from primary key fields")
    composite_key_components: Dict[str, Any] = Field(description="Individual components of composite key")
    
    # Location in data
    nbim_record_identifier: Optional[str] = Field(None, description="Identifier for NBIM record(s)")
    custody_record_identifier: Optional[str] = Field(None, description="Identifier for Custody record(s)")
    affected_field: Optional[str] = Field(None, description="Field(s) where discrepancy found")
    
    # Values comparison
    nbim_value: Optional[Any] = Field(None, description="Value from NBIM")
    custody_value: Optional[Any] = Field(None, description="Value from Custody")
    difference: Optional[Any] = Field(None, description="Calculated difference if applicable")
    
    # Impact and confidence
    impact_assessment: Optional[str] = Field(None, description="Assessment of impact")
    confidence: float = Field(ge=0, le=1, description="Confidence in this identification")


class BreaksAnalysisResponse(BaseAgentResponse):
    """Response from the breaks identification agent."""
    breaks: List[BreakIdentification] = Field(description="List of identified breaks")
    
    # Summary statistics (let LLM determine what's important)
    total_breaks_found: int = Field(description="Total number of breaks identified")
    classification_summary: Dict[str, int] = Field(description="Count of breaks by classification")
    severity_summary: Dict[str, int] = Field(description="Count of breaks by severity")
    
    # Overall assessment
    overall_assessment: str = Field(description="Overall assessment of data reconciliation")
    data_quality_observations: List[str] = Field(description="General observations about data quality")
    
    # Metadata
    total_nbim_records_analyzed: int = Field(description="Number of NBIM records analyzed")
    total_custody_records_analyzed: int = Field(description="Number of Custody records analyzed")
    analysis_approach: str = Field(description="Brief description of analysis approach used")


class BreaksIdentifierAgent:
    """
    Pure LLM-driven agent for identifying breaks in reconciliation.
    No hardcoded categories - lets the LLM determine what's important.
    Uses composite keys for unique record identification.
    """
    
    def __init__(self):
        self.agent_name = "breaks_identifier_agent"
        # Define primary key fields that form the composite key
        self.primary_key_fields = [
            'coac_event_key',
            'isin', 
            'sedol',
            'bank_account'
        ]
    
    def _build_composite_key(self, record: Dict[str, Any], key_fields: List[str]) -> str:
        """
        Build a composite key from primary key fields.
        
        Args:
            record: The data record
            key_fields: List of field names to use for composite key
            
        Returns:
            Composite key string
        """
        key_values = []
        for field in key_fields:
            # Try different case variations
            value = None
            for key in record.keys():
                if key.lower().replace('_', '').replace(' ', '') == field.lower().replace('_', '').replace(' ', ''):
                    value = record[key]
                    break
            
            if value is not None:
                key_values.append(str(value))
            else:
                key_values.append('NULL')
        
        return '|'.join(key_values)
    
    def _extract_composite_key_components(self, record: Dict[str, Any], key_fields: List[str]) -> Dict[str, Any]:
        """Extract the actual values for composite key fields from a record."""
        components = {}
        for field in key_fields:
            # Try different case variations
            value = None
            actual_field = None
            for key in record.keys():
                if key.lower().replace('_', '').replace(' ', '') == field.lower().replace('_', '').replace(' ', ''):
                    value = record[key]
                    actual_field = key
                    break
            
            if actual_field:
                components[field] = value
            else:
                components[field] = None
                
        return components

    def _build_system_prompt(self) -> str:
        """Build system prompt that guides LLM without constraining it."""
        return """You are an expert financial reconciliation analyst specializing in dividend data analysis.

Your task is to compare two datasets (NBIM and Custody) and identify ALL discrepancies/breaks between them.

CRITICAL: Use COMPOSITE KEYS for record matching and unique identification:
- The composite key consists of: coac_event_key, isin, sedol, bank_account
- These fields together form a unique identifier for each record
- Match records between datasets using these composite keys
- Each break should be associated with its composite key for uniqueness

Guidelines:
1. BUILD composite keys from the primary key fields (coac_event_key, isin, sedol, bank_account)
2. MATCH records between datasets using composite keys
3. IDENTIFY any differences, missing records, or inconsistencies
4. CLASSIFY each break based on what you observe (you determine the categories)
5. ASSIGN severity based on your assessment of business impact
6. EXPLAIN each break clearly and concisely
7. BE THOROUGH - catch everything from major discrepancies to minor formatting issues

You should:
- Use composite keys to match corresponding records between datasets
- Report missing records based on composite key mismatches
- Compare field values only for records with matching composite keys
- Identify value differences in mapped fields
- Notice date discrepancies
- Detect calculation errors
- Find any other anomalies or issues

For each break, provide:
- The composite key (concatenated with '|' separator)
- The individual composite key components
- A clear classification (you decide the category names based on what you find)
- Severity level (you decide the scale based on impact)
- Confidence in your identification (0-1)
- Impact assessment

Return structured JSON following the BreaksAnalysisResponse schema."""
    
    def _build_analysis_prompt(
        self,
        nbim_data: List[Dict[str, Any]],
        custody_data: List[Dict[str, Any]],
        mappings: Optional[Dict[str, str]] = None,
        primary_keys: Optional[List[str]] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """Build the analysis prompt with data."""
        
        # Use provided primary keys or default ones
        pk_fields = primary_keys or self.primary_key_fields
        
        # Pre-compute composite keys for the data
        nbim_with_keys = []
        for record in nbim_data[:50]:  # Limit for prompt
            composite_key = self._build_composite_key(record, pk_fields)
            enriched_record = {**record, '_composite_key': composite_key}
            nbim_with_keys.append(enriched_record)
            
        custody_with_keys = []
        for record in custody_data[:50]:  # Limit for prompt
            composite_key = self._build_composite_key(record, pk_fields)
            enriched_record = {**record, '_composite_key': composite_key}
            custody_with_keys.append(enriched_record)
        
        prompt = f"""Please analyze these two datasets and identify all breaks/discrepancies using composite keys:

PRIMARY KEY FIELDS (Use these to build composite keys):
{json.dumps(pk_fields, indent=2)}

NBIM DATA ({len(nbim_data)} records):
```json
{json.dumps(nbim_with_keys, indent=2, default=str)}  
```
{f'... and {len(nbim_data) - 50} more records' if len(nbim_data) > 50 else ''}

CUSTODY DATA ({len(custody_data)} records):
```json
{json.dumps(custody_with_keys, indent=2, default=str)}
```
{f'... and {len(custody_data) - 50} more records' if len(custody_data) > 50 else ''}
"""

        if mappings:
            nbim_fields = list(mappings.keys())
            custody_fields = list(mappings.values())
            prompt += f"""

FIELD MAPPINGS (NBIM field -> Custody field):
```json
{json.dumps(mappings, indent=2)}
```

IMPORTANT INSTRUCTIONS:
- Only compare the fields defined in the mappings above.
- Ignore and do not report any differences for fields that are NOT listed in these mappings.
- For each reported break, the affected_field MUST be one of these mapped NBIM fields:
```json
{json.dumps(nbim_fields, indent=2)}
```
If a discrepancy involves a Custody field, it must correspond to the mapped Custody field for the same NBIM field.
"""

        if additional_context:
            prompt += f"""

ADDITIONAL CONTEXT:
{additional_context}
"""

        prompt += """

Please perform a comprehensive reconciliation analysis using COMPOSITE KEYS:

1. Use the _composite_key field (or build your own from primary key fields) to match records
2. Identify records that exist in one dataset but not the other (based on composite keys)
3. For matching composite keys, compare field values to find discrepancies
4. Classify each break based on its nature (you determine the classifications)
5. Assess severity based on business impact (you determine the severity scale)
6. Provide your confidence level for each identified break

IMPORTANT: Each break must include:
- composite_key: The concatenated key using '|' separator (e.g., "EVT001|US123|SEDOL123|ACC001")
- composite_key_components: Object with individual key values

Return your analysis as JSON matching this structure:

{
  "success": true,
  "breaks": [
    {
      "break_id": "BRK_001",
      "description": "Clear description of what's wrong",
      "classification": "Your classification for this type of break",
      "severity": "Your severity assessment",
      "composite_key": "EVT001|US123|SEDOL123|ACC001",
      "composite_key_components": {
        "coac_event_key": "EVT001",
        "isin": "US123",
        "sedol": "SEDOL123",
        "bank_account": "ACC001"
      },
      "nbim_record_identifier": "identifier from NBIM record",
      "custody_record_identifier": "identifier from Custody record",
      "affected_field": "field name where issue found",
      "nbim_value": "value from NBIM",
      "custody_value": "value from Custody",
      "difference": "calculated difference if applicable",
      "impact_assessment": "Your assessment of the impact",
      "confidence": 0.95
    }
  ],
  "total_breaks_found": 5,
  "classification_summary": {
    "your_category_1": 2,
    "your_category_2": 3
  },
  "severity_summary": {
    "your_severity_1": 1,
    "your_severity_2": 4
  },
  "overall_assessment": "Your overall assessment of the reconciliation",
  "data_quality_observations": [
    "Observation 1",
    "Observation 2"
  ],
  "total_nbim_records_analyzed": 100,
  "total_custody_records_analyzed": 98,
  "analysis_approach": "Brief description of how you analyzed the data using composite keys"
}

Be thorough and catch everything. Each break should be uniquely identified by its composite key."""
        
        return prompt
    
    async def identify_breaks(
        self,
        nbim_data: List[Dict[str, Any]],
        custody_data: List[Dict[str, Any]],
        mappings: Optional[Dict[str, str]] = None,
        primary_keys: Optional[List[str]] = None,
        additional_context: Optional[str] = None,
        timeout: float = 180.0,
        temperature: float = 0.1
    ) -> AgentResult:
        """
        Identify breaks between NBIM and Custody data using pure LLM analysis with composite keys.
        
        Args:
            nbim_data: List of NBIM records
            custody_data: List of Custody records  
            mappings: Field mappings from mapping agent
            primary_keys: List of fields that form the composite key
            additional_context: Any additional context for the analysis
            timeout: API timeout
            temperature: LLM temperature (lower = more consistent)
            
        Returns:
            AgentResult with identified breaks using composite keys
        """
        
        logger.info(f"Starting LLM-driven breaks analysis: {len(nbim_data)} NBIM, {len(custody_data)} Custody records")
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt()
            },
            {
                "role": "user",
                "content": self._build_analysis_prompt(
                    nbim_data, custody_data, mappings, primary_keys, additional_context
                )
            }
        ]
        
        # Execute agent
        executor = AgentExecutor()
        result = await executor.execute_single_agent(
            agent_name=self.agent_name,
            messages=messages,
            expected_response_type=BreaksAnalysisResponse,
            timeout=timeout,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        return result
    
    async def identify_breaks_in_batches(
        self,
        nbim_data: List[Dict[str, Any]],
        custody_data: List[Dict[str, Any]],
        mappings: Optional[Dict[str, str]] = None,
        primary_keys: Optional[List[str]] = None,
        batch_size: int = 100,
        additional_context: Optional[str] = None
    ) -> AgentResult:
        """
        Process large datasets in batches to avoid token limits.
        
        Args:
            nbim_data: List of NBIM records
            custody_data: List of Custody records
            mappings: Field mappings
            batch_size: Number of records per batch
            additional_context: Additional context
            
        Returns:
            Combined AgentResult
        """
        
        all_breaks = []
        classification_summary = {}
        severity_summary = {}
        
        # Process in batches
        for i in range(0, max(len(nbim_data), len(custody_data)), batch_size):
            nbim_batch = nbim_data[i:i+batch_size]
            custody_batch = custody_data[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}")
            
            result = await self.identify_breaks(
                nbim_batch,
                custody_batch,
                mappings,
                primary_keys,
                f"{additional_context}\nBatch {i//batch_size + 1}" if additional_context else f"Batch {i//batch_size + 1}"
            )
            
            if result.status == AgentStatus.COMPLETED and result.response.structured_data:
                batch_data = result.response.structured_data
                all_breaks.extend(batch_data.get("breaks", []))
                
                # Aggregate summaries
                for cls, count in batch_data.get("classification_summary", {}).items():
                    classification_summary[cls] = classification_summary.get(cls, 0) + count
                
                for sev, count in batch_data.get("severity_summary", {}).items():
                    severity_summary[sev] = severity_summary.get(sev, 0) + count
        
        # Create combined response
        combined_response = {
            "success": True,
            "breaks": all_breaks,
            "total_breaks_found": len(all_breaks),
            "classification_summary": classification_summary,
            "severity_summary": severity_summary,
            "overall_assessment": f"Analyzed {len(nbim_data)} NBIM and {len(custody_data)} Custody records in batches",
            "data_quality_observations": [],
            "total_nbim_records_analyzed": len(nbim_data),
            "total_custody_records_analyzed": len(custody_data),
            "analysis_approach": f"Batch processing with size {batch_size}"
        }
        
        # Create result
        from utils.response_parser import ParsedResponse
        return AgentResult(
            agent_name=self.agent_name,
            status=AgentStatus.COMPLETED,
            response=ParsedResponse(
                content="Batch processing complete",
                structured_data=combined_response
            )
        )
    
    def extract_unique_classifications(self, result: AgentResult) -> Tuple[List[str], List[str]]:
        """
        Extract the unique classifications and severities that the LLM identified.
        
        Returns:
            Tuple of (classifications list, severities list)
        """
        
        if not result.response or not result.response.structured_data:
            return [], []
        
        breaks = result.response.structured_data.get("breaks", [])
        
        classifications = list(set(b.get("classification", "Unknown") for b in breaks))
        severities = list(set(b.get("severity", "Unknown") for b in breaks))
        
        return sorted(classifications), sorted(severities)
    
    def get_breaks_by_classification(self, result: AgentResult, classification: str) -> List[Dict[str, Any]]:
        """Get all breaks of a specific classification."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        breaks = result.response.structured_data.get("breaks", [])
        return [b for b in breaks if b.get("classification") == classification]
    
    def get_breaks_by_severity(self, result: AgentResult, severity: str) -> List[Dict[str, Any]]:
        """Get all breaks of a specific severity."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        breaks = result.response.structured_data.get("breaks", [])
        return [b for b in breaks if b.get("severity") == severity]
    
    def get_breaks_by_composite_key(self, result: AgentResult, composite_key: str) -> List[Dict[str, Any]]:
        """Get all breaks for a specific composite key."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        breaks = result.response.structured_data.get("breaks", [])
        return [b for b in breaks if b.get("composite_key") == composite_key]
    
    def get_unique_composite_keys(self, result: AgentResult) -> List[str]:
        """Get all unique composite keys from the breaks."""
        
        if not result.response or not result.response.structured_data:
            return []
        
        breaks = result.response.structured_data.get("breaks", [])
        composite_keys = set(b.get("composite_key", "") for b in breaks if b.get("composite_key"))
        return sorted(list(composite_keys))
    
    def group_breaks_by_composite_key(self, result: AgentResult) -> Dict[str, List[Dict[str, Any]]]:
        """Group breaks by their composite keys."""
        
        if not result.response or not result.response.structured_data:
            return {}
        
        breaks = result.response.structured_data.get("breaks", [])
        grouped = {}
        
        for break_item in breaks:
            key = break_item.get("composite_key", "NO_KEY")
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(break_item)
        
        return grouped

    def validate_breaks(self, result: AgentResult) -> bool:
        """Basic validation for LLM-driven analysis output."""
        if result.status != AgentStatus.COMPLETED:
            logger.error(f"Agent failed: {result.error}")
            return False
        if not result.response or not result.response.structured_data:
            logger.error("No structured data in response")
            return False
        data = result.response.structured_data
        if "breaks" not in data or "total_breaks_found" not in data:
            logger.error("Missing required fields in response")
            return False
        return True


# Example usage
async def example_usage():
    """Example of using the pure LLM-driven breaks identifier."""
    
    # Sample NBIM data with all primary key fields
    nbim_data = [
        {
            "coac_event_key": "EVT001",
            "isin": "US0378331005",
            "sedol": "2046251",
            "bank_account": "ACC12345",
            "security_name": "Apple Inc",
            "ex_date": "2024-02-09",
            "pay_date": "2024-02-15",
            "quantity": 10000,
            "dividend_rate": 0.24,
            "gross_amount": 2400.00,
            "withholding_tax": 360.00,
            "net_amount": 2040.00,
            "currency": "USD",
            "record_date": "2024-02-08"
        },
        {
            "coac_event_key": "EVT002",
            "isin": "US5949181045",
            "sedol": "2588173",
            "bank_account": "ACC12345",
            "security_name": "Microsoft Corp",
            "ex_date": "2024-02-14",
            "pay_date": "2024-03-14",
            "quantity": 5000,
            "dividend_rate": 0.75,
            "gross_amount": 3750.00,
            "withholding_tax": 562.50,
            "net_amount": 3187.50,
            "currency": "USD",
            "record_date": "2024-02-13"
        },
        {
            "coac_event_key": "EVT003",
            "isin": "US0378331005",
            "sedol": "2046251",
            "bank_account": "ACC67890",  # Different account for same security
            "security_name": "Apple Inc",
            "ex_date": "2024-02-09",
            "pay_date": "2024-02-15",
            "quantity": 5000,
            "dividend_rate": 0.24,
            "gross_amount": 1200.00,
            "withholding_tax": 180.00,
            "net_amount": 1020.00,
            "currency": "USD",
            "record_date": "2024-02-08"
        }
    ]
    
    # Sample Custody data (with intentional discrepancies for testing)
    custody_data = [
        {
            "event_id": "EVT001",
            "isin": "US0378331005",
            "sedol_code": "2046251",
            "account_number": "ACC12345",
            "instrument": "Apple Inc",
            "ex_dividend_date": "2024-02-09",
            "payment_date": "2024-02-15",
            "shares": 10000,
            "rate_per_share": 0.24,
            "gross_dividend": 2400.00,
            "tax_withheld": 300.00,  # Different tax amount
            "net_dividend": 2100.00,  # Different net amount
            "ccy": "USD",
            "record_dt": "2024-02-08"
        },
        {
            "event_id": "EVT003",
            "isin": "US0378331005",
            "sedol_code": "2046251",
            "account_number": "ACC67890",
            "instrument": "Apple Inc",
            "ex_dividend_date": "2024-02-09",
            "payment_date": "2024-02-15",
            "shares": 5500,  # Different quantity
            "rate_per_share": 0.24,
            "gross_dividend": 1320.00,  # Different gross
            "tax_withheld": 198.00,
            "net_dividend": 1122.00,
            "ccy": "USD",
            "record_dt": "2024-02-08"
        }
        # Missing Microsoft record (EVT002)
    ]
    
    # Field mappings from mapping agent
    mappings = {
        "coac_event_key": "event_id",
        "isin": "isin",
        "sedol": "sedol_code",
        "bank_account": "account_number",
        "security_name": "instrument",
        "ex_date": "ex_dividend_date",
        "pay_date": "payment_date",
        "quantity": "shares",
        "dividend_rate": "rate_per_share",
        "gross_amount": "gross_dividend",
        "withholding_tax": "tax_withheld",
        "net_amount": "net_dividend",
        "currency": "ccy",
        "record_date": "record_dt"
    }
    
    # Define primary keys for composite key
    primary_keys = ["coac_event_key", "isin", "sedol", "bank_account"]
    
    # Additional context
    context = """
    This is dividend reconciliation data for Q1 2024.
    NBIM is the internal booking system.
    Custody is the external custodian data.
    Tax rates should be 15% for US securities.
    All amounts are in the security's local currency.
    """
    
    # Create and run the agent
    agent = BreaksIdentifierAgent()
    
    print("🔍 Running LLM-driven breaks identification with composite keys...")
    result = await agent.identify_breaks(
        nbim_data=nbim_data,
        custody_data=custody_data,
        mappings=mappings,
        primary_keys=primary_keys,
        additional_context=context
    )
    
    # Process results
    if result.status == AgentStatus.COMPLETED and result.response.structured_data:
        data = result.response.structured_data
        
        print(f"\n✅ Analysis complete!")
        print(f"\n📊 Summary:")
        print(f"  Total breaks found: {data.get('total_breaks_found', 0)}")
        
        # Show classifications that LLM came up with
        print(f"\n📁 Classifications identified by LLM:")
        for cls, count in data.get('classification_summary', {}).items():
            print(f"  - {cls}: {count}")
        
        # Show severities that LLM assigned
        print(f"\n⚠️ Severities assigned by LLM:")
        for sev, count in data.get('severity_summary', {}).items():
            print(f"  - {sev}: {count}")
        
        # Show first few breaks with composite keys
        print(f"\n🔍 Sample breaks identified (with composite keys):")
        for break_item in data.get('breaks', [])[:5]:
            print(f"\n  [{break_item.get('severity', 'N/A').upper()}] {break_item.get('classification', 'N/A')}")
            print(f"  🔑 Composite Key: {break_item.get('composite_key', 'N/A')}")
            if 'composite_key_components' in break_item:
                components = break_item['composite_key_components']
                print(f"     Components: CoAC={components.get('coac_event_key', 'N/A')}, "
                      f"ISIN={components.get('isin', 'N/A')}, "
                      f"SEDOL={components.get('sedol', 'N/A')}, "
                      f"Account={components.get('bank_account', 'N/A')}")
            print(f"  Description: {break_item.get('description', 'N/A')}")
            print(f"  Impact: {break_item.get('impact_assessment', 'N/A')}")
            print(f"  Confidence: {break_item.get('confidence', 0):.0%}")
        
        # Show unique composite keys
        unique_keys = agent.get_unique_composite_keys(result)
        print(f"\n🔑 Unique Composite Keys Found: {len(unique_keys)}")
        for key in unique_keys[:3]:
            print(f"  - {key}")
        
        # Group breaks by composite key
        grouped_breaks = agent.group_breaks_by_composite_key(result)
        print(f"\n📊 Breaks grouped by composite key:")
        for key, breaks_list in list(grouped_breaks.items())[:3]:
            print(f"  Key: {key}")
            print(f"    - {len(breaks_list)} break(s) found")
            for b in breaks_list:
                print(f"      • {b.get('classification', 'N/A')}: {b.get('affected_field', 'N/A')}")
        
        print(f"\n💡 Overall Assessment:")
        print(f"  {data.get('overall_assessment', 'N/A')}")
        
        print(f"\n📝 Data Quality Observations:")
        for obs in data.get('data_quality_observations', []):
            print(f"  - {obs}")
        
        # Extract unique classifications and severities
        classifications, severities = agent.extract_unique_classifications(result)
        print(f"\n🏷️ Unique classifications: {', '.join(classifications)}")
        print(f"🎯 Unique severities: {', '.join(severities)}")
        
    else:
        print(f"❌ Analysis failed: {result.error if result.error else 'Unknown error'}")


if __name__ == "__main__":
    asyncio.run(example_usage())
