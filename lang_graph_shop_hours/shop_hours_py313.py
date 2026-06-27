# shop_hours_working.py
"""
Shop Hours Assistant - Windows Compatible (No Emojis)
Works with Python 3.13
"""
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# ============== FIX WINDOWS ENCODING ==============
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============== VERSION CHECK ==============
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

# ============== TRY IMPORTS WITH FALLBACKS ==============

LANGCHAIN_AVAILABLE = False
try:
    # Try newer import paths first
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import PydanticOutputParser
    except ImportError:
        # Fallback to older paths
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.output_parsers import PydanticOutputParser
    
    from pydantic import BaseModel, Field
    from langgraph.graph import StateGraph, END
    LANGCHAIN_AVAILABLE = True
    print("[OK] LangChain loaded successfully")
except ImportError as e:
    print(f"[INFO] LangChain not available: {e}")
    print("[INFO] Running in mock data mode only")

# ============== MOCK DATA ==============

MOCK_HOURS = {
    "starbucks": {
        "Monday": "6:00 AM - 8:00 PM",
        "Tuesday": "6:00 AM - 8:00 PM",
        "Wednesday": "6:00 AM - 8:00 PM",
        "Thursday": "6:00 AM - 8:00 PM",
        "Friday": "6:00 AM - 9:00 PM",
        "Saturday": "7:00 AM - 9:00 PM",
        "Sunday": "7:00 AM - 7:00 PM"
    },
    "walmart": {
        "Monday": "6:00 AM - 11:00 PM",
        "Tuesday": "6:00 AM - 11:00 PM",
        "Wednesday": "6:00 AM - 11:00 PM",
        "Thursday": "6:00 AM - 11:00 PM",
        "Friday": "6:00 AM - 11:00 PM",
        "Saturday": "6:00 AM - 11:00 PM",
        "Sunday": "6:00 AM - 11:00 PM"
    },
    "target": {
        "Monday": "8:00 AM - 10:00 PM",
        "Tuesday": "8:00 AM - 10:00 PM",
        "Wednesday": "8:00 AM - 10:00 PM",
        "Thursday": "8:00 AM - 10:00 PM",
        "Friday": "8:00 AM - 10:00 PM",
        "Saturday": "8:00 AM - 10:00 PM",
        "Sunday": "8:00 AM - 9:00 PM"
    },
    "costco": {
        "Monday": "10:00 AM - 8:30 PM",
        "Tuesday": "10:00 AM - 8:30 PM",
        "Wednesday": "10:00 AM - 8:30 PM",
        "Thursday": "10:00 AM - 8:30 PM",
        "Friday": "10:00 AM - 8:30 PM",
        "Saturday": "9:30 AM - 6:00 PM",
        "Sunday": "Closed"
    },
    "best buy": {
        "Monday": "10:00 AM - 8:00 PM",
        "Tuesday": "10:00 AM - 8:00 PM",
        "Wednesday": "10:00 AM - 8:00 PM",
        "Thursday": "10:00 AM - 8:00 PM",
        "Friday": "10:00 AM - 8:00 PM",
        "Saturday": "10:00 AM - 8:00 PM",
        "Sunday": "11:00 AM - 7:00 PM"
    },
    "home depot": {
        "Monday": "6:00 AM - 10:00 PM",
        "Tuesday": "6:00 AM - 10:00 PM",
        "Wednesday": "6:00 AM - 10:00 PM",
        "Thursday": "6:00 AM - 10:00 PM",
        "Friday": "6:00 AM - 10:00 PM",
        "Saturday": "6:00 AM - 10:00 PM",
        "Sunday": "8:00 AM - 8:00 PM"
    },
    "lowes": {
        "Monday": "6:00 AM - 10:00 PM",
        "Tuesday": "6:00 AM - 10:00 PM",
        "Wednesday": "6:00 AM - 10:00 PM",
        "Thursday": "6:00 AM - 10:00 PM",
        "Friday": "6:00 AM - 10:00 PM",
        "Saturday": "6:00 AM - 10:00 PM",
        "Sunday": "8:00 AM - 8:00 PM"
    }
}

# ============== CORE FUNCTIONS ==============

def find_shop(query: str) -> Optional[str]:
    """Find shop name in query"""
    query_lower = query.lower()
    for shop in MOCK_HOURS.keys():
        if shop in query_lower:
            return shop
    return None

def get_today() -> str:
    """Get today's day name"""
    return datetime.now().strftime("%A")

def format_hours_response(shop_name: str, hours: Dict[str, str]) -> str:
    """Format hours nicely"""
    today = get_today()
    
    lines = []
    lines.append("=" * 50)
    lines.append(f"  {shop_name.title()} Opening Hours")
    lines.append("=" * 50)
    lines.append("")
    
    max_len = max(len(day) for day in hours.keys())
    
    for day, time in hours.items():
        marker = ">>" if day == today else "  "
        lines.append(f"{marker} {day.ljust(max_len)} : {time}")
    
    lines.append("")
    lines.append(f"[Today: {today}]")
    lines.append(f"  {hours.get(today, 'Unknown')}")
    lines.append("=" * 50)
    
    return "\n".join(lines)

def get_mock_hours(query: str) -> str:
    """Get hours from mock data"""
    shop_name = find_shop(query)
    
    if not shop_name:
        available = ", ".join(MOCK_HOURS.keys())
        return f"\n[ERROR] Shop not found.\nAvailable shops: {available}\n"
    
    return format_hours_response(shop_name, MOCK_HOURS[shop_name])

# ============== LLM VERSION (if available) ==============

if LANGCHAIN_AVAILABLE:
    try:
        class ShopHoursLLM:
            def __init__(self, api_key: Optional[str] = None):
                self.api_key = api_key or os.getenv("OPENAI_API_KEY")
                self.use_llm = bool(self.api_key)
                
                if self.use_llm:
                    try:
                        self.llm = ChatOpenAI(
                            model="gpt-3.5-turbo",
                            temperature=0,
                            api_key=self.api_key
                        )
                        print("[OK] LLM mode enabled")
                    except Exception as e:
                        print(f"[WARN] LLM init failed: {e}")
                        self.use_llm = False
            
            def get_hours(self, query: str) -> str:
                # First try mock data
                shop_name = find_shop(query)
                if shop_name:
                    return format_hours_response(shop_name, MOCK_HOURS[shop_name])
                
                # If not in mock and LLM available, try LLM
                if self.use_llm:
                    try:
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "You are a helpful assistant that provides shop hours."),
                            ("user", f"Provide the opening hours for the shop mentioned in: {query}. Today is {get_today()}")
                        ])
                        
                        chain = prompt | self.llm
                        response = chain.invoke({})
                        return f"\n[LLM Response]\n{response.content}\n"
                    except Exception as e:
                        return f"\n[ERROR] LLM error: {e}\n"
                
                return f"\n[ERROR] Shop not found.\nAvailable shops: {', '.join(MOCK_HOURS.keys())}\n"
    
    except Exception as e:
        print(f"[WARN] LLM setup error: {e}")

# ============== INTERACTIVE MODE ==============

def interactive_mode():
    """Interactive CLI"""
    print("\n" + "=" * 50)
    print("  SHOP HOURS ASSISTANT")
    print("=" * 50)
    print(f"  Available shops: {', '.join(MOCK_HOURS.keys())}")
    print("  Type 'exit' to quit")
    print("=" * 50)
    
    # Setup LLM if available
    llm_handler = None
    if LANGCHAIN_AVAILABLE:
        try:
            llm_handler = ShopHoursLLM()
        except Exception as e:
            print(f"[WARN] LLM setup error: {e}")
    
    while True:
        query = input("\n[?] Ask about shop hours: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        print("\n[SEARCHING] ...")
        
        if llm_handler and llm_handler.use_llm:
            result = llm_handler.get_hours(query)
        else:
            result = get_mock_hours(query)
        
        print(result)
        print("-" * 50)

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = get_mock_hours(query)
        print(result)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
