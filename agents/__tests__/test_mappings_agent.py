"""
Test script for the mappings agent.
Run this to test your OpenAI integration before integrating with Streamlit.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the path so we can import agents and utils
sys.path.append(os.path.dirname(__file__))

from agents.mappings_agent import MappingsAgent, example_usage
from utils import logger, config


async def test_setup():
    """Test that everything is set up correctly."""
    
    print("🔧 Testing OpenAI setup...")
    
    # Check if API key is set
    if not config.openai.api_key:
        print("❌ OPENAI_API_KEY not found!")
        print("Please create a .env file with your API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ API key found (starts with: {config.openai.api_key[:12]}...)")
    print(f"✅ Model: {config.openai.model}")
    print(f"✅ Temperature: {config.openai.temperature}")
    
    return True


async def test_simple_mapping():
    """Test with simple, obvious mappings."""
    
    print("\n🧪 Testing simple mappings...")
    
    simple_source = ["ID", "Name", "Amount"] 
    simple_target = ["Identifier", "Description", "Value"]
    
    agent = MappingsAgent()
    result = await agent.create_mappings(
        source_headers=simple_source,
        target_headers=simple_target,
        context="Simple test data with obvious mappings"
    )
    
    if agent.validate_mappings(result):
        print("✅ Simple mapping test passed!")
        data = result.response.structured_data
        print(f"   Mappings found: {data.get('total_mappings', 0)}")
        for mapping in data.get('mappings', []):
            print(f"   {mapping['source_header']} → {mapping['target_header']}")
    else:
        print("❌ Simple mapping test failed!")
        if result.error:
            print(f"   Error: {result.error}")
    
    return result


async def main():
    """Main test function."""
    
    print("🚀 Testing Mappings Agent Setup\n")
    
    # Test setup
    if not await test_setup():
        return
    
    # Test simple mapping
    await test_simple_mapping()
    
    # Test complex example
    print("\n🧪 Testing complex financial data example...")
    await example_usage()
    
    print("\n✅ All tests completed!")
    print("\n📝 Next steps:")
    print("1. Check the logs/ directory for detailed logs")
    print("2. Integrate with your Streamlit app using integrate_with_streamlit()")
    print("3. Create more agents following this same pattern")


if __name__ == "__main__":
    asyncio.run(main())