# shop_hours_groq.py
"""
Shop Hours Assistant - Using Groq (Free API)
Sign up at: https://console.groq.com
"""
import os
from datetime import datetime
from typing import Optional

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# ============== GROQ SETUP ==============
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[INFO] Groq not installed. Install with: pip install groq")

# Get Groq API key (FREE)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
    },
    "mcdonalds": {
        "Monday": "5:00 AM - 11:00 PM",
        "Tuesday": "5:00 AM - 11:00 PM",
        "Wednesday": "5:00 AM - 11:00 PM",
        "Thursday": "5:00 AM - 11:00 PM",
        "Friday": "5:00 AM - 12:00 AM",
        "Saturday": "5:00 AM - 12:00 AM",
        "Sunday": "6:00 AM - 11:00 PM"
    },
    "subway": {
        "Monday": "8:00 AM - 10:00 PM",
        "Tuesday": "8:00 AM - 10:00 PM",
        "Wednesday": "8:00 AM - 10:00 PM",
        "Thursday": "8:00 AM - 10:00 PM",
        "Friday": "8:00 AM - 10:00 PM",
        "Saturday": "8:00 AM - 10:00 PM",
        "Sunday": "9:00 AM - 9:00 PM"
    }
}

# ============== HELPER FUNCTIONS ==============

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

def format_mock_response(shop_name: str) -> str:
    """Format mock response"""
    today = get_today()
    hours = MOCK_HOURS[shop_name]
    
    lines = []
    lines.append("=" * 50)
    lines.append(f"  {shop_name.title()} Opening Hours")
    lines.append("=" * 50)
    lines.append("")
    
    for day, time in hours.items():
        marker = ">>" if day == today else "  "
        lines.append(f"{marker} {day}: {time}")
    
    lines.append("")
    lines.append(f"[Today: {today}]")
    lines.append(f"  {hours[today]}")
    lines.append("=" * 50)
    
    return "\n".join(lines)

# ============== GROQ LLM FUNCTION (FREE) ==============

def get_groq_hours(query: str, api_key: str) -> str:
    """Get hours using Groq API (FREE)"""
    if not GROQ_AVAILABLE:
        return "[ERROR] Groq not installed. Install with: pip install groq"
    
    if not api_key:
        return "[ERROR] No Groq API key. Get free key at: https://console.groq.com"
    
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""Provide the opening hours for the shop mentioned in this query: {query}
        
        Today is {get_today()}.
        
        Format your response clearly with:
        - Days of the week (Monday through Sunday)
        - Opening and closing times for each day
        - Mark today's hours with [TODAY]
        - Note if the shop is closed on certain days
        
        If you don't know the exact hours, provide typical hours for that type of shop.
        """
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",  # Free model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate shop opening hours."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        response = completion.choices[0].message.content
        
        return f"\n[Groq AI Response - FREE]\n{'-' * 50}\n{response}\n{'-' * 50}\n"
        
    except Exception as e:
        return f"[ERROR] Groq API error: {str(e)}"

# ============== MAIN FUNCTION ==============

def get_shop_hours(query: str, api_key: Optional[str] = None) -> str:
    """Main function to get shop hours"""
    # First check if shop is in mock data
    shop_name = find_shop(query)
    
    if shop_name:
        return format_mock_response(shop_name)
    
    # If not in mock data and we have Groq API key
    if api_key and GROQ_AVAILABLE:
        print("[INFO] Shop not in mock data, trying Groq AI...")
        return get_groq_hours(query, api_key)
    
    # Fallback
    available = ", ".join(MOCK_HOURS.keys())
    return f"\n[INFO] Shop not found in database.\nAvailable shops: {available}\n"

# ============== INTERACTIVE MODE ==============

def interactive_mode():
    """Interactive CLI"""
    print("\n" + "=" * 50)
    print("  SHOP HOURS ASSISTANT")
    print("=" * 50)
    print(f"  Available shops: {', '.join(MOCK_HOURS.keys())}")
    
    # Check API status
    if GROQ_API_KEY and GROQ_AVAILABLE:
        print("  [OK] Groq AI: ENABLED (FREE)")
    else:
        print("  [INFO] Groq AI: DISABLED")
        print("  [INFO] Get free key at: https://console.groq.com")
    
    print("  Type 'exit' to quit")
    print("=" * 50)
    
    while True:
        query = input("\n[?] Ask about shop hours: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        print("\n[SEARCHING] ...")
        result = get_shop_hours(query, GROQ_API_KEY)
        print(result)
        print("-" * 50)

# ============== MAIN ==============

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = get_shop_hours(query, GROQ_API_KEY)
        print(result)
    else:
        interactive_mode()
