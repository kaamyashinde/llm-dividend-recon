"""Response parsing and validation utilities."""

import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass

from .logger import logger
from .error_handler import ValidationError as AgentValidationError

T = TypeVar('T', bound=BaseModel)


@dataclass
class ParsedResponse:
    """Container for parsed agent responses."""
    content: str
    structured_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


class ResponseParser:
    """Utility class for parsing and validating agent responses."""
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text that may contain other content."""
        # Try to find JSON within code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON without code blocks
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    @staticmethod
    def extract_list_from_text(text: str) -> Optional[List[str]]:
        """Extract list items from text."""
        # Look for numbered lists
        numbered_pattern = r'^\d+\.\s*(.+)$'
        numbered_matches = re.findall(numbered_pattern, text, re.MULTILINE)
        
        if numbered_matches:
            return numbered_matches
        
        # Look for bullet points
        bullet_pattern = r'^[-*•]\s*(.+)$'
        bullet_matches = re.findall(bullet_pattern, text, re.MULTILINE)
        
        if bullet_matches:
            return bullet_matches
        
        # Look for lines starting with dashes
        dash_pattern = r'^-\s*(.+)$'
        dash_matches = re.findall(dash_pattern, text, re.MULTILINE)
        
        if dash_matches:
            return dash_matches
        
        return None
    
    @classmethod
    def parse_response(
        self,
        response_content: str,
        expected_format: Optional[Type[T]] = None,
        agent_name: str = "unknown"
    ) -> ParsedResponse:
        """Parse agent response with optional validation."""
        
        parsed = ParsedResponse(content=response_content)
        
        try:
            # Try to extract structured data
            json_data = self.extract_json_from_text(response_content)
            if json_data:
                parsed.structured_data = json_data
            
            # Try to extract lists if no JSON found
            if not json_data:
                list_data = self.extract_list_from_text(response_content)
                if list_data:
                    parsed.structured_data = {"items": list_data}
            
            # Validate against expected format if provided
            if expected_format and parsed.structured_data:
                try:
                    validated_data = expected_format(**parsed.structured_data)
                    parsed.structured_data = validated_data.dict()
                except ValidationError as e:
                    error_details = []
                    for error in e.errors():
                        field = " -> ".join(str(x) for x in error["loc"])
                        message = error["msg"]
                        error_details.append(f"{field}: {message}")
                    
                    parsed.validation_errors.extend(error_details)
                    logger.warning(
                        f"Validation failed for {agent_name}: {'; '.join(error_details)}"
                    )
            
            # Extract metadata
            parsed.metadata = {
                "has_structured_data": parsed.structured_data is not None,
                "has_validation_errors": len(parsed.validation_errors) > 0,
                "content_length": len(response_content),
                "agent_name": agent_name
            }
            
        except Exception as e:
            error_msg = f"Error parsing response from {agent_name}: {str(e)}"
            parsed.validation_errors.append(error_msg)
            logger.error(error_msg)
        
        return parsed
    
    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any], 
        required_fields: List[str],
        agent_name: str = "unknown"
    ) -> List[str]:
        """Validate that required fields are present in the data."""
        errors = []
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif data[field] is None or data[field] == "":
                errors.append(f"Empty required field: {field}")
        
        if errors:
            logger.warning(f"Required field validation failed for {agent_name}: {'; '.join(errors)}")
        
        return errors
    
    @staticmethod
    def clean_response_text(text: str) -> str:
        """Clean up response text by removing unnecessary whitespace and formatting."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove code block markers if they wrap the entire response
        if text.strip().startswith('```') and text.strip().endswith('```'):
            lines = text.strip().split('\n')
            if len(lines) > 2:
                # Remove first and last line if they're just code block markers
                if lines[0].strip().startswith('```') and lines[-1].strip() == '```':
                    text = '\n'.join(lines[1:-1])
        
        return text.strip()


# Common response models for validation
class BaseAgentResponse(BaseModel):
    """Base class for agent responses."""
    success: bool = True
    message: Optional[str] = None
    confidence: Optional[float] = None


class ListResponse(BaseAgentResponse):
    """Response containing a list of items."""
    items: List[str]


class AnalysisResponse(BaseAgentResponse):
    """Response for analysis tasks."""
    findings: List[str]
    recommendations: List[str]
    summary: str


class ValidationResponse(BaseAgentResponse):
    """Response for validation tasks."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class ReportResponse(BaseAgentResponse):
    """Response for report generation."""
    title: str
    sections: List[Dict[str, Any]]
    conclusion: str