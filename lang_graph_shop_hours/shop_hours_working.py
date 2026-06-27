# shop_hours_working.py
import sys
import io
import os
from datetime import datetime
from typing import TypedDict, Optional, Dict, List, Any

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Try importing required packages
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    from langgraph.graph import StateGraph, END
    PACKAGES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Some packages are missing: {e}")
    print("\n📦 Please install required packages:")
    print("  pip install langchain langchain-openai langgraph pydantic python-dotenv")
    print("\n🔄 Falling back to mock data mode...")
    PACKAGES_AVAILABLE = False

# ============== Data Models (if packages available) ==============

if PACKAGES_AVAILABLE:
    class ShopHours(BaseModel):
        shop_name: str = Field(description="Name of the shop")
        day: str = Field(description="Day of the week")
        opens_at: str = Field(description="Opening time in HH:MM format")
        closes_at: str = Field(description="Closing time in HH:MM format")
        is_closed: bool = Field(default=False)
        special_note: Optional[str] = Field(default=None)

    class ShopHoursResponse(BaseModel):
        shop_name: str = Field(description="Name of the shop")
        hours: List[ShopHours] = Field(description="List of daily hours")
        is_24_hours: bool = Field(default=False)
        last_updated: str = Field(description="When this information was last updated")

# ============== Mock Data ==============

MOCK_HOURS = {
    "starbucks": {
        "Monday": "6:00 - 20:00",
        "Tuesday": "6:00 - 20:00",
        "Wednesday": "6:00 - 20:00",
        "Thursday": "6:00 - 20:00",
        "Friday": "6:00 - 21:00",
        "Saturday": "7:00 - 21:00",
        "Sunday": "7:00 - 19:00"
    },
    "walmart": {
        "Monday": "6:00 - 23:00",
        "Tuesday": "6:00 - 23:00",
        "Wednesday": "6:00 - 23:00",
        "Thursday": "6:00 - 23:00",
        "Friday": "6:00 - 23:00",
        "Saturday": "6:00 - 23:00",
        "Sunday": "6:00 - 23:00"
    },
    "target": {
        "Monday": "8:00 - 22:00",
        "Tuesday": "8:00 - 22:00",
        "Wednesday": "8:00 - 22:00",
        "Thursday": "8:00 - 22:00",
        "Friday": "8:00 - 22:00",
        "Saturday": "8:00 - 22:00",
        "Sunday": "8:00 - 21:00"
    },
    "costco": {
        "Monday": "10:00 - 20:30",
        "Tuesday": "10:00 - 20:30",
        "Wednesday": "10:00 - 20:30",
        "Thursday": "10:00 - 20:30",
        "Friday": "10:00 - 20:30",
        "Saturday": "9:30 - 18:00",
        "Sunday": "Closed"
    },
    "pharmacy": {
        "Monday": "9:00 - 21:00",
        "Tuesday": "9:00 - 21:00",
        "Wednesday": "9:00 - 21:00",
        "Thursday": "9:00 - 21:00",
        "Friday": "9:00 - 21:00",
        "Saturday": "9:00 - 18:00",
        "Sunday": "10:00 - 17:00"
    }
}

# ============== State ==============

class ShopHoursState(TypedDict):
    query: str
    shop_name: Optional[str]
    current_day: Optional[str]
    final_response: Optional[str]
    error: Optional[str]

# ============== Core Functions ==============

def extract_shop_name(query: str) -> Optional[str]:
    """Extract shop name from query"""
    query_lower = query.lower()
    
    # Check for shop names in query
    for shop in MOCK_HOURS.keys():
        if shop in query_lower:
            return shop
    
    return None

def get_shop_hours_mock(shop_name: str, current_day: str) -> str:
    """Get shop hours from mock data"""
    if shop_name not in MOCK_HOURS:
        return f"Shop '{shop_name}' not found. Available shops: {', '.join(MOCK_HOURS.keys())}"
    
    hours = MOCK_HOURS[shop_name]
    
    # Format response
    response = f"\n{'='*50}\n"
    response += f"{shop_name.title()} Opening Hours\n"
    response += f"{'='*50}\n\n"
    
    for day, time in hours.items():
        marker = "►" if day == current_day else "  "
        response += f"{marker} {day}: {time}\n"
    
    response += f"\nToday: {current_day} - {hours.get(current_day, 'Unknown')}"
    response += f"\n{'='*50}"
    
    return response

def run_shop_hours(query: str) -> str:
    """Main function to get shop hours"""
    current_day = datetime.now().strftime("%A")
    shop_name = extract_shop_name(query)
    
    if not shop_name:
        return f"\n❌ Could not find shop in your query.\nAvailable shops: {', '.join(MOCK_HOURS.keys())}\n"
    
    return get_shop_hours_mock(shop_name, current_day)

# ============== LLM Version (if packages available) ==============

if PACKAGES_AVAILABLE:
    class LLMShopHours:
        def __init__(self, api_key: str = None):
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.use_llm = bool(self.api_key)
            
            if self.use_llm:
                try:
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0,
                        api_key=self.api_key
                    )
                    self.parser = PydanticOutputParser(pydantic_object=ShopHoursResponse)
                    print("✅ LLM mode enabled")
                except Exception as e:
                    print(f"⚠️  LLM initialization failed: {e}")
                    print("🔄 Falling back to mock data")
                    self.use_llm = False
        
        def get_hours(self, query: str) -> str:
            current_day = datetime.now().strftime("%A")
            shop_name = extract_shop_name(query)
            
            if not shop_name:
                return f"\n❌ Could not find shop in your query.\nAvailable shops: {', '.join(MOCK_HOURS.keys())}\n"
            
            if not self.use_llm:
                return get_shop_hours_mock(shop_name, current_day)
            
            # Try LLM
            try:
                prompt = ChatPromptTemplate.from_template(
                    """Provide the opening hours for {shop_name} in JSON format.
                    Current day is: {current_day}
                    
                    Return a JSON object with shop_name and hours array.
                    Each hour entry should have: day, opens_at, closes_at, is_closed.
                    
                    Shop: {shop_name}
                    """
                )
                
                format_instructions = self.parser.get_format_instructions()
                chain = prompt | self.llm
                response = chain.invoke({
                    "shop_name": shop_name,
                    "current_day": current_day,
                })
                
                return f"LLM Response: {response.content}"
                
            except Exception as e:
                print(f"⚠️  LLM failed: {e}")
                return get_shop_hours_mock(shop_name, current_day)

# ============== Interactive Mode ==============

def interactive_mode():
    """Run in interactive mode"""
    print("\n" + "="*50)
    print(" SHOP HOURS ASSISTANT ")
    print("="*50)
    print(f"Available shops: {', '.join(MOCK_HOURS.keys())}")
    print("Type 'exit' to quit")
    print("="*50)
    
    # Try to initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and PACKAGES_AVAILABLE:
        print("\n💡 To use LLM mode, set OPENAI_API_KEY environment variable")
        print("   Currently using mock data mode\n")
    
    llm_handler = LLMShopHours(api_key) if PACKAGES_AVAILABLE else None
    
    while True:
        query = input("\n🔍 Ask about shop hours: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        print("\n⏳ Searching...")
        
        if llm_handler:
            result = llm_handler.get_hours(query)
        else:
            result = run_shop_hours(query)
        
        print(result)
        print("-"*50)

# ============== Main ==============

if __name__ == "__main__":
    # Check if running with query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = run_shop_hours(query)
        print(result)
    else:
        interactive_mode()
