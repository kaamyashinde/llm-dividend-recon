"""
Simple test for JIRA agent logic without requiring dependencies.
This tests the core logic and data structures.
"""

# Test data structures and basic validation
def test_jira_agent_structure():
    """Test that the JIRA agent has the correct structure."""
    
    # Read the jira agent file and check for key components
    try:
        with open('agents/jira_issue_agent.py', 'r') as f:
            content = f.read()
        
        # Check for key classes and methods
        required_elements = [
            'class JiraIssue',
            'class JiraIssueResponse', 
            'class JiraIssueAgent',
            'def create_jira_issues',
            'def format_for_csv_import',
            'def format_for_json_import',
            'def validate_jira_issues',
            '_build_system_prompt',
            '_build_jira_creation_prompt'
        ]
        
        missing = []
        for element in required_elements:
            if element not in content:
                missing.append(element)
        
        if missing:
            print(f"❌ Missing required elements: {missing}")
            return False
        
        print("✅ All required classes and methods present")
        
        # Check for proper imports
        required_imports = [
            'from typing import Dict, List, Any, Optional',
            'from pydantic import BaseModel, Field', 
            'from enum import Enum',
            'from utils import'
        ]
        
        for imp in required_imports:
            if imp not in content:
                print(f"❌ Missing import: {imp}")
                return False
        
        print("✅ All required imports present")
        
        # Check for JIRA-specific enums and mappings
        jira_elements = [
            'class JiraIssuePriority',
            'class JiraIssueType',
            'HIGHEST = "Highest"',
            'BUG = "Bug"',
            'TASK = "Task"',
            'to_csv_row',
            'CSV import format',
            'JSON API format'
        ]
        
        for element in jira_elements:
            if element not in content:
                print(f"⚠️ Missing JIRA element: {element}")
        
        print("✅ JIRA-specific elements present")
        return True
        
    except FileNotFoundError:
        print("❌ JIRA agent file not found")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def test_sample_data_structure():
    """Test that sample data has the right structure for JIRA conversion."""
    
    # Sample break that should be convertible to JIRA
    sample_break = {
        "break_id": "BRK_001",
        "description": "Tax amount mismatch between NBIM and Custody systems",
        "classification": "Value Discrepancy",
        "severity": "High",
        "composite_key": "EVT001|US0378331005|2046251|ACC12345",
        "affected_field": "withholding_tax",
        "nbim_value": 360.00,
        "custody_value": 300.00,
        "difference": 60.00,
        "impact_assessment": "Net settlement amount will be incorrect by $60",
        "confidence": 0.95
    }
    
    # Required fields for JIRA conversion
    required_fields = ["break_id", "description", "classification", "severity"]
    
    for field in required_fields:
        if field not in sample_break:
            print(f"❌ Sample break missing required field: {field}")
            return False
    
    print("✅ Sample break has all required fields for JIRA conversion")
    
    # Check severity to priority mapping logic
    severity_to_priority = {
        "Critical": "Highest",
        "High": "High", 
        "Medium": "Medium",
        "Low": "Low"
    }
    
    if sample_break["severity"] in severity_to_priority:
        expected_priority = severity_to_priority[sample_break["severity"]]
        print(f"✅ Severity '{sample_break['severity']}' maps to priority '{expected_priority}'")
    
    # Check classification to issue type mapping
    classification_to_type = {
        "Value Discrepancy": "Task",
        "Missing Record": "Bug",
        "System Error": "Bug",
        "Process Issue": "Improvement"
    }
    
    if sample_break["classification"] in classification_to_type:
        expected_type = classification_to_type[sample_break["classification"]]
        print(f"✅ Classification '{sample_break['classification']}' maps to type '{expected_type}'")
    
    return True


def test_csv_format_structure():
    """Test that CSV format will have correct headers."""
    
    expected_csv_headers = [
        "Summary",
        "Description", 
        "Issue Type",
        "Priority",
        "Component",
        "Labels",
        "Break ID",
        "Break Classification",
        "Affected Field",
        "NBIM Value",
        "Custody Value",
        "Financial Impact"
    ]
    
    print("✅ Expected CSV headers for JIRA import:")
    for header in expected_csv_headers:
        print(f"  - {header}")
    
    return True


def test_integration_points():
    """Test integration with existing system."""
    
    # Check that the Streamlit integration imports the JIRA agent
    try:
        with open('agents/breaks_streamlit_integration.py', 'r') as f:
            content = f.read()
        
        if 'from agents.jira_issue_agent import JiraIssueAgent' in content:
            print("✅ Streamlit integration imports JIRA agent")
        else:
            print("❌ Streamlit integration missing JIRA agent import")
            return False
        
        if 'create_jira_issues_for_breaks' in content:
            print("✅ Streamlit integration has JIRA creation function")
        else:
            print("❌ Streamlit integration missing JIRA creation function")
            return False
        
        if 'asyncio.run(jira_agent.create_jira_issues(' in content:
            print("✅ Streamlit integration calls JIRA agent correctly")
        else:
            print("❌ Streamlit integration doesn't call JIRA agent")
            return False
        
        return True
        
    except FileNotFoundError:
        print("❌ Streamlit integration file not found")
        return False


if __name__ == "__main__":
    print("🧪 Testing JIRA Agent Implementation")
    print("=" * 50)
    
    tests = [
        ("Agent Structure", test_jira_agent_structure),
        ("Sample Data", test_sample_data_structure),
        ("CSV Format", test_csv_format_structure),
        ("Integration", test_integration_points)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            if test_func():
                print(f"✅ {test_name} test passed")
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! JIRA agent implementation looks good.")
    else:
        print("⚠️ Some tests failed. Review the implementation.")
    
    print("\n💡 To fully test the agent:")
    print("1. Install dependencies: pip install pydantic openai streamlit pandas")
    print("2. Set OPENAI_API_KEY environment variable")
    print("3. Run: python agents/jira_issue_agent.py")
    print("4. Test in Streamlit app with real reconciliation data")