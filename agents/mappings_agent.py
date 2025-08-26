"""
Mappings Agent - Maps table headers between different data sources.

This agent analyzes table headers from different sources (e.g., NBIM and Custody files)
and creates mappings between them for data reconciliation.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import asyncio

# Import utils (assuming utils package is in the same directory level)
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


# Define the expected response structure
class HeaderMapping(BaseModel):
    """Single header mapping between source and target."""
    source_header: str = Field(description="Original header name")
    target_header: str = Field(description="Mapped/standardized header name")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)
    reasoning: str = Field(description="Explanation for the mapping")
    is_primary_key: bool = Field(default=False, description="Whether this column is a primary key")


class MappingsResponse(BaseAgentResponse):
    """Response structure for mappings agent."""
    mappings: List[HeaderMapping] = Field(description="List of header mappings")
    unmapped_headers: List[str] = Field(default=[], description="Headers that couldn't be mapped")
    primary_keys: List[str] = Field(default=[], description="List of identified primary key columns")
    summary: str = Field(description="Summary of the mapping process")
    total_mappings: int = Field(description="Total number of successful mappings")


class MappingsAgent:
    """Agent for creating header mappings between data sources."""
    
    def __init__(self):
        self.agent_name = "mappings_agent"
        
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the mappings agent."""
        return """You are a data mapping expert specializing in financial data reconciliation. 
        
Your task is to create mappings between table headers from different data sources. You will be given:
1. Source headers (e.g., from NBIM files)
2. Target headers (e.g., from Custody files)
3. Context about what the data represents

Your job is to:
- Match similar headers between sources
- Standardize header names for consistency
- Provide confidence scores for your mappings
- Explain your reasoning for each mapping
- Identify headers that cannot be mapped
- IMPORTANT: Identify primary key columns

PRIMARY KEY IDENTIFICATION:
The following columns should ALWAYS be identified as primary keys when found:
- coac_event_key (or variations like COAC_EVENT_KEY, CoAcEventKey, etc.)
- isin (or variations like ISIN, Isin, ISIN_Code, etc.)
- sedol (or variations like SEDOL, Sedol, SEDOL_Code, etc.)  
- bank account (or variations like bank_account, BankAccount, Bank_Account, Account_Number, etc.)

These columns represent unique identifiers and should be marked with is_primary_key: true in the mapping.

Guidelines:
- Look for semantic similarity, not just exact matches
- Consider common abbreviations and variations (e.g., "Qty" = "Quantity")
- Financial terms may have standard mappings (e.g., "ISIN" = "Security ID")
- Be conservative with confidence scores - only high confidence (>0.8) for obvious matches
- Always identify primary key columns (coac_event_key, isin, sedol, bank account)
- Always return valid JSON in the specified format

Return your response as JSON matching the MappingsResponse schema."""

    def _build_user_prompt(
        self, 
        source_headers: List[str], 
        target_headers: List[str],
        context: Optional[str] = None
    ) -> str:
        """Build the user prompt with the actual data."""
        
        prompt = f"""Please create mappings between these table headers:

SOURCE HEADERS:
{chr(10).join(f"- {header}" for header in source_headers)}

TARGET HEADERS:
{chr(10).join(f"- {header}" for header in target_headers)}"""

        if context:
            prompt += f"""

CONTEXT:
{context}"""

        prompt += """

Please analyze these headers and create mappings where appropriate. 
IMPORTANT: Identify primary key columns (coac_event_key, isin, sedol, bank account or their variations).

Return your response as JSON following this exact structure:

{
  "success": true,
  "mappings": [
    {
      "source_header": "original_header_name",
      "target_header": "mapped_header_name", 
      "confidence": 0.95,
      "reasoning": "Explanation for this mapping",
      "is_primary_key": false
    }
  ],
  "unmapped_headers": ["header1", "header2"],
  "primary_keys": ["isin", "sedol", "coac_event_key", "bank_account"],
  "summary": "Brief summary of mapping process including identified primary keys",
  "total_mappings": 5
}"""

        return prompt

    async def create_mappings(
        self,
        source_headers: List[str],
        target_headers: List[str],
        context: Optional[str] = None,
        timeout: float = 60.0
    ) -> AgentResult:
        """
        Create header mappings between source and target headers.
        
        Args:
            source_headers: List of source table headers
            target_headers: List of target table headers  
            context: Optional context about the data
            timeout: Timeout for the API call
            
        Returns:
            AgentResult with the mapping response
        """
        
        logger.info(f"Starting header mapping: {len(source_headers)} source, {len(target_headers)} target headers")
        
        # Build the messages for OpenAI
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt()
            },
            {
                "role": "user", 
                "content": self._build_user_prompt(source_headers, target_headers, context)
            }
        ]
        
        # Execute the agent
        executor = AgentExecutor()
        result = await executor.execute_single_agent(
            agent_name=self.agent_name,
            messages=messages,
            expected_response_type=MappingsResponse,
            timeout=timeout,
            # OpenAI specific parameters
            temperature=0.1,  # Low temperature for consistent results
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        return result
    
    def validate_mappings(self, result: AgentResult) -> bool:
        """
        Validate that the mappings result is usable.
        
        Args:
            result: The agent result to validate
            
        Returns:
            True if the result is valid and usable
        """
        
        if result.status != AgentStatus.COMPLETED:
            logger.error(f"Agent failed: {result.error}")
            return False
            
        if not result.response or not result.response.structured_data:
            logger.error("No structured data in response")
            return False
            
        if result.response.validation_errors:
            logger.warning(f"Validation errors: {result.response.validation_errors}")
            return False
            
        return True
    
    def get_primary_keys(self, result: AgentResult) -> List[str]:
        """
        Extract primary key columns from the mapping result.
        
        Args:
            result: The agent result containing mappings
            
        Returns:
            List of identified primary key column names
        """
        
        if not self.validate_mappings(result):
            return []
        
        primary_keys = []
        mappings_data = result.response.structured_data
        
        # Get from primary_keys field
        primary_keys.extend(mappings_data.get('primary_keys', []))
        
        # Also check individual mappings for is_primary_key flag
        for mapping in mappings_data.get('mappings', []):
            if mapping.get('is_primary_key', False):
                # Add both source and target headers as primary keys
                if mapping.get('source_header') and mapping['source_header'] not in primary_keys:
                    primary_keys.append(mapping['source_header'])
                if mapping.get('target_header') and mapping['target_header'] not in primary_keys:
                    primary_keys.append(mapping['target_header'])
        
        # Ensure we always include the core primary keys if found in any form
        core_keys = ['coac_event_key', 'isin', 'sedol', 'bank_account']
        for key in core_keys:
            # Check if any header contains the core key (case-insensitive)
            for header in mappings_data.get('mappings', []):
                source = header.get('source_header', '').lower()
                target = header.get('target_header', '').lower()
                key_lower = key.replace('_', '').replace(' ', '').lower()
                
                if key_lower in source.replace('_', '').replace(' ', '').lower():
                    if header['source_header'] not in primary_keys:
                        primary_keys.append(header['source_header'])
                if key_lower in target.replace('_', '').replace(' ', '').lower():
                    if header['target_header'] not in primary_keys:
                        primary_keys.append(header['target_header'])
        
        return primary_keys


# Example usage functions
async def example_usage():
    """Example of how to use the mappings agent."""
    
    # Sample data (you would get this from your CSV files)
    nbim_headers = [
        "coac_event_key",
        "Security_ID", 
        "Security_Name", 
        "ISIN",
        "SEDOL",
        "Bank_Account",
        "Quantity", 
        "Unit_Price",
        "Market_Value_NOK",
        "Currency_Code"
    ]
    
    custody_headers = [
        "COAC_EVENT_KEY",
        "ISIN",
        "Instrument_Description",
        "SEDOL_Code", 
        "Account_Number",
        "Qty",
        "Price_Per_Unit",
        "Market_Val_Local",
        "Currency"
    ]
    
    context = """
    This is financial data for dividend reconciliation. 
    NBIM headers are from the Norwegian Bank Investment Management files.
    Custody headers are from the custody bank files.
    Both contain information about securities, quantities, and market values.
    """
    
    # Create and run the agent
    agent = MappingsAgent()
    result = await agent.create_mappings(
        source_headers=nbim_headers,
        target_headers=custody_headers,
        context=context
    )
    
    # Check if successful
    if agent.validate_mappings(result):
        mappings_data = result.response.structured_data
        print("✅ Mappings created successfully!")
        print(f"Total mappings: {mappings_data.get('total_mappings', 0)}")
        
        # Print primary keys first
        primary_keys = mappings_data.get('primary_keys', [])
        if primary_keys:
            print(f"\n🔑 Primary Keys Identified: {', '.join(primary_keys)}")
        
        # Print each mapping
        print("\n📊 Header Mappings:")
        for mapping in mappings_data.get('mappings', []):
            pk_indicator = " 🔑" if mapping.get('is_primary_key', False) else ""
            print(f"  {mapping['source_header']} → {mapping['target_header']}{pk_indicator} "
                  f"(confidence: {mapping['confidence']:.2f})")
            print(f"    Reasoning: {mapping['reasoning']}")
        
        # Print unmapped headers
        unmapped = mappings_data.get('unmapped_headers', [])
        if unmapped:
            print(f"\n⚠️ Unmapped headers: {unmapped}")
            
    else:
        print("❌ Failed to create mappings")
        if result.error:
            print(f"Error: {result.error}")


# Streamlit integration helper
def integrate_with_streamlit(nbim_df, custody_df):
    """
    Helper function to integrate with your Streamlit app.
    Call this from your app.py after loading the dataframes.
    """
    
    nbim_headers = list(nbim_df.columns)
    custody_headers = list(custody_df.columns)
    
    # Run the mapping agent
    async def run_mapping():
        agent = MappingsAgent()
        result = await agent.create_mappings(
            source_headers=nbim_headers,
            target_headers=custody_headers,
            context="Financial dividend reconciliation data"
        )
        return result
    
    # Note: In Streamlit, you'll need to handle async differently
    # You might need to use asyncio.run() or similar
    return asyncio.run(run_mapping())


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())