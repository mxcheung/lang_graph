# First, install required packages
# pip install langgraph langchain-openai python-dotenv

import os
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
import json
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ============== 1. Define Data Models ==============

class ShopHours(BaseModel):
    """Structure for shop opening hours"""
    shop_name: str = Field(description="Name of the shop")
    day: str = Field(description="Day of the week")
    opens_at: str = Field(description="Opening time in HH:MM format")
    closes_at: str = Field(description="Closing time in HH:MM format")
    is_closed: bool = Field(default=False, description="Whether the shop is closed on this day")
    special_note: Optional[str] = Field(default=None, description="Any special notes about hours")

class ShopHoursResponse(BaseModel):
    """Full response containing shop hours"""
    shop_name: str = Field(description="Name of the shop")
    hours: List[ShopHours] = Field(description="List of daily hours")
    is_24_hours: bool = Field(default=False, description="Whether shop is open 24/7")
    last_updated: str = Field(description="When this information was last updated")

# ============== 2. Define State ==============

class ShopHoursState(TypedDict):
    """State management for shop hours workflow"""
    query: str  # User's query about shop hours
    shop_name: Optional[str]  # Extracted shop name
    current_day: Optional[str]  # Current day of week
    raw_llm_response: Optional[str]  # Raw LLM response
    parsed_hours: Optional[Dict]  # Structured hours data
    final_response: Optional[str]  # Final formatted response
    error: Optional[str]  # Error messages
    needs_clarification: bool  # Whether we need more info

# ============== 3. Define Nodes ==============

class ShopHoursNodes:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = PydanticOutputParser(pydantic_object=ShopHoursResponse)
        
    def extract_shop_name(self, state: ShopHoursState) -> ShopHoursState:
        """Extract shop name from the query"""
        query = state["query"]
        
        # Simple extraction - in production, use NER or LLM
        prompt = ChatPromptTemplate.from_template(
            """Extract the shop name from this query. Return only the shop name.
            If no shop is mentioned, return "unknown".
            
            Query: {query}
            
            Shop name:"""
        )
        
        chain = prompt | self.llm
        shop_name = chain.invoke({"query": query}).content.strip()
        
        state["shop_name"] = shop_name if shop_name != "unknown" else None
        state["needs_clarification"] = shop_name == "unknown"
        
        return state
    
    def get_current_day(self, state: ShopHoursState) -> ShopHoursState:
        """Get current day of week"""
        state["current_day"] = datetime.now().strftime("%A")
        return state
    
    def query_llm_for_hours(self, state: ShopHoursState) -> ShopHoursState:
        """Query LLM for shop hours"""
        if state.get("error"):
            return state
            
        shop_name = state.get("shop_name")
        current_day = state.get("current_day")
        
        if not shop_name:
            state["error"] = "No shop name provided"
            return state
        
        # Create prompt for getting shop hours
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that provides shop opening hours.
            Provide the opening hours for {shop_name} in the following JSON format:
            {format_instructions}
            
            Current day is: {current_day}
            
            If you don't know the exact hours, provide your best estimate based on common
            business hours for this type of shop. Mark days as closed if appropriate.
            
            Shop: {shop_name}
            Hours:"""
        )
        
        format_instructions = self.parser.get_format_instructions()
        
        chain = prompt | self.llm
        response = chain.invoke({
            "shop_name": shop_name,
            "current_day": current_day,
            "format_instructions": format_instructions
        })
        
        state["raw_llm_response"] = response.content
        
        # Parse the response
        try:
            # Clean up the response - remove markdown code blocks if present
            cleaned_response = response.content
            if "```json" in cleaned_response:
                cleaned_response = cleaned_response.split("```json")[1].split("```")[0]
            elif "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[1].split("```")[0]
            
            parsed = self.parser.parse(cleaned_response)
            state["parsed_hours"] = parsed.dict()
            state["needs_clarification"] = False
        except Exception as e:
            state["error"] = f"Failed to parse hours: {str(e)}"
            state["needs_clarification"] = True
        
        return state
    
    def format_response(self, state: ShopHoursState) -> ShopHoursState:
        """Format the final response"""
        if state.get("error"):
            state["final_response"] = f"Error: {state['error']}"
            return state
        
        if state.get("needs_clarification"):
            state["final_response"] = "I need more information. Which shop are you asking about?"
            return state
        
        parsed = state.get("parsed_hours")
        if not parsed:
            state["final_response"] = "Unable to retrieve shop hours at this time."
            return state
        
        # Format the response nicely
        shop_name = parsed.get("shop_name", "Unknown Shop")
        hours_data = parsed.get("hours", [])
        current_day = state.get("current_day", "Unknown")
        is_24_hours = parsed.get("is_24_hours", False)
        
        if is_24_hours:
            response = f"🏪 {shop_name} is open 24/7!"
        else:
            response = f"🏪 {shop_name} Opening Hours:\n\n"
            
            # Group hours by day
            for day_hours in hours_data:
                day = day_hours.get("day", "Unknown Day")
                is_closed = day_hours.get("is_closed", False)
                opens = day_hours.get("opens_at", "N/A")
                closes = day_hours.get("closes_at", "N/A")
                note = day_hours.get("special_note")
                
                if is_closed:
                    response += f"• {day}: Closed\n"
                else:
                    response += f"• {day}: {opens} - {closes}\n"
                
                if note:
                    response += f"  Note: {note}\n"
        
        # Add current day info
        response += f"\n📅 Today is {current_day}"
        
        # Find today's hours
        today_hours = next((h for h in hours_data if h.get("day") == current_day), None)
        if today_hours:
            if today_hours.get("is_closed"):
                response += " - Closed today"
            else:
                response += f" - Open {today_hours.get('opens_at')} to {today_hours.get('closes_at')}"
        
        state["final_response"] = response
        return state
    
    def handle_error(self, state: ShopHoursState) -> ShopHoursState:
        """Handle errors gracefully"""
        if state.get("error"):
            state["final_response"] = f"I encountered an error: {state['error']}. Please try again."
        return state

# ============== 4. Build the Graph ==============

def build_shop_hours_graph():
    """Build the LangGraph workflow"""
    # Initialize nodes
    nodes = ShopHoursNodes()
    
    # Create the graph
    workflow = StateGraph(ShopHoursState)
    
    # Add nodes
    workflow.add_node("extract_shop", nodes.extract_shop_name)
    workflow.add_node("get_day", nodes.get_current_day)
    workflow.add_node("query_llm", nodes.query_llm_for_hours)
    workflow.add_node("format_response", nodes.format_response)
    workflow.add_node("handle_error", nodes.handle_error)
    
    # Add edges
    workflow.set_entry_point("extract_shop")
    
    # Extract shop -> get current day -> query LLM
    workflow.add_edge("extract_shop", "get_day")
    workflow.add_edge("get_day", "query_llm")
    
    # Conditional edge from query_llm
    def should_format(state: ShopHoursState) -> str:
        if state.get("error"):
            return "handle_error"
        return "format_response"
    
    workflow.add_conditional_edges(
        "query_llm",
        should_format,
        {
            "format_response": "format_response",
            "handle_error": "handle_error"
        }
    )
    
    # Format response -> END
    workflow.add_edge("format_response", END)
    workflow.add_edge("handle_error", END)
    
    # Compile the graph
    return workflow.compile()

# ============== 5. Usage Example ==============

def get_shop_hours(query: str) -> str:
    """Main function to get shop hours"""
    # Initialize graph
    graph = build_shop_hours_graph()
    
    # Initial state
    initial_state = {
        "query": query,
        "shop_name": None,
        "current_day": None,
        "raw_llm_response": None,
        "parsed_hours": None,
        "final_response": None,
        "error": None,
        "needs_clarification": False
    }
    
    # Run the graph
    try:
        final_state = graph.invoke(initial_state)
        return final_state.get("final_response", "No response generated")
    except Exception as e:
        return f"Error processing request: {str(e)}"

# ============== 6. Example Usage ==============

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What are the opening hours for Starbucks?",
        "When does Walmart open today?",
        "Is Target open on Sundays?",
        "What time does the local pharmacy close?",
    ]
    
    for query in test_queries:
        print("=" * 60)
        print(f"Query: {query}")
        print("-" * 60)
        result = get_shop_hours(query)
        print(result)
        print()
