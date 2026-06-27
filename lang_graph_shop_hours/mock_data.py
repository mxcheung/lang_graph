# mock_data.py - Fixed version without emojis
import json
from typing import TypedDict, Optional
from datetime import datetime

class ShopHoursState(TypedDict):
    query: str
    shop_name: Optional[str]
    current_day: Optional[str]
    final_response: Optional[str]
    error: Optional[str]

# Mock shop hours database
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
    }
}

def extract_shop_name(state: ShopHoursState) -> ShopHoursState:
    """Extract shop name from query"""
    query_lower = state["query"].lower()
    
    for shop in MOCK_HOURS.keys():
        if shop in query_lower:
            state["shop_name"] = shop
            return state
    
    state["shop_name"] = None
    state["error"] = "Shop not found in database"
    return state

def get_current_day(state: ShopHoursState) -> ShopHoursState:
    """Get current day"""
    state["current_day"] = datetime.now().strftime("%A")
    return state

def get_shop_hours(state: ShopHoursState) -> ShopHoursState:
    """Get shop hours from mock database"""
    if state.get("error"):
        return state
    
    shop_name = state.get("shop_name")
    current_day = state.get("current_day")
    
    if not shop_name or shop_name not in MOCK_HOURS:
        state["error"] = f"No information available for {shop_name}"
        return state
    
    hours = MOCK_HOURS[shop_name]
    today_hours = hours.get(current_day, "Unknown")
    
    # Format response WITHOUT emojis
    response = f"[SHOP] {shop_name.title()} Opening Hours:\n\n"
    for day, time in hours.items():
        response += f"  * {day}: {time}\n"
    
    response += f"\n[Today] ({current_day}): {today_hours}"
    
    state["final_response"] = response
    return state

def run_shop_hours_pipeline(query: str) -> str:
    """Run the simplified pipeline"""
    state = {
        "query": query,
        "shop_name": None,
        "current_day": None,
        "final_response": None,
        "error": None
    }
    
    state = extract_shop_name(state)
    state = get_current_day(state)
    state = get_shop_hours(state)
    
    if state.get("error"):
        return f"Error: {state['error']}"
    
    return state.get("final_response", "No response")

# Test the simplified version
if __name__ == "__main__":
    queries = [
        "What are Starbucks hours?",
        "When does Walmart open?",
        "Target hours please",
        "Local bookstore hours"
    ]
    
    for query in queries:
        print("=" * 50)
        print(f"Q: {query}")
        print(run_shop_hours_pipeline(query))
        print()
