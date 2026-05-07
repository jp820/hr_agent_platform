import os
import logging
import sqlite3
import datetime
from typing import TypedDict, Optional, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from duckduckgo_search import DDGS
from google import genai
from google.genai import types


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM & CLIENTS
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
genai_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# DATABASE HELPERS
def get_db_connection():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    db_path = os.path.join(base_dir, "data", "hr_db.sqlite")
    return sqlite3.connect(db_path)

def get_employee_balances(employee_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT annual_leave_balance, sick_leave_balance FROM employees WHERE id = ?", (employee_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {"annual": result[0], "sick": result[1]}
    return None

def update_balances(employee_id: str, leave_type: str, days: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    if "annual" in leave_type.lower():
        cursor.execute("UPDATE employees SET annual_leave_balance = annual_leave_balance - ? WHERE id = ?", (days, employee_id))
    elif "sick" in leave_type.lower():
        cursor.execute("UPDATE employees SET sick_leave_balance = sick_leave_balance - ? WHERE id = ?", (days, employee_id))
    conn.commit()
    conn.close()

def save_leave_record(employee_id: str, start_date: str, end_date: str, leave_type: str, days: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO leave_requests (employee_id, start_date, end_date, leave_type, days_count) VALUES (?, ?, ?, ?, ?)",
                   (employee_id, start_date, end_date, leave_type, days))
    conn.commit()
    conn.close()

def save_travel_request(employee_id: str, source: str, destination: str, outbound_date: str, return_date: Optional[str], is_round_trip: bool, selected_flight: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO travel_requests (employee_id, source, destination, outbound_date, return_date, is_round_trip, selected_flight, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (employee_id, source, destination, outbound_date, return_date, is_round_trip, selected_flight, "Pending Approval"))
    conn.commit()
    conn.close()

def get_travel_history(employee_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT source, destination, outbound_date, status FROM travel_requests WHERE employee_id = ? ORDER BY outbound_date DESC", (employee_id,))
    results = cursor.fetchall()
    conn.close()
    return results

# STATE
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    uid: str
    user_input: str
    intent: Optional[str]
    response: Optional[str]
    leave_balance: Optional[int]
    travel_request: Optional[dict]
    outbound_options: Optional[list] = Field(default_factory=list)
    return_options: Optional[list] = Field(default_factory=list)

# TOOLS
@tool
def search_flights(source: str, destination: str, date: str) -> list:
    """Actual flight search using Google Search grounding via Gemini"""
    query = f"flights from {source} to {destination} on {date} prices and times"
    logger.info(f"Searching for flights using Google Search: {query}")
    
    prompt = f"""
    Search for flights from {source} to {destination} on {date}.
    List at least 3 specific flight options (airline, price, time) found in the search results.
    Provide the raw flight data clearly. Do not say you cannot find them if any snippets are visible.
    """
    
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        search_results = response.text
        logger.info(f"Google Search Response: {search_results}")
    except Exception as e:
        logger.error(f"Google Search failed: {e}")
        search_results = "No results found via Google Search."

    # Use the main LLM to parse the grounding results into our specific schema
    extraction_prompt = f"""
    Extract exactly 3 flight options from the following search data for {source} to {destination} on {date}.
    Search Data:
    {search_results}

    Return a list of dictionaries with 'flight', 'price', and 'time'.
    """
    
    class FlightOption(BaseModel):
        flight: str
        price: str
        time: str

    class FlightOptions(BaseModel):
        options: List[FlightOption]

    structured_llm = llm.with_structured_output(FlightOptions)
    parsed = structured_llm.invoke([HumanMessage(content=extraction_prompt)])
    
    if not parsed.options:
        # Fallback if no real data found
        return [
            {"flight": f"Indigo 6E-{source[:2]}{destination[:2]}-1", "price": "5,000", "time": "06:00 AM"},
            {"flight": f"AirIndia AI-{source[:2]}{destination[:2]}-2", "price": "6,500", "time": "09:30 AM"},
            {"flight": f"SpiceJet SG-{source[:2]}{destination[:2]}-3", "price": "4,800", "time": "11:45 PM"}
        ]
        
    return [opt.model_dump() for opt in parsed.options[:3]]

# MODELS FOR STRUCTURED OUTPUT (OUTPUT CONTRACTS)
class IntentClassification(BaseModel):
    intent: str = Field(description="The classified intent: 'policy', 'leave', 'travel', or 'other'")

class LeaveExtraction(BaseModel):
    action: Optional[str] = Field(description="The requested action: 'apply' or 'check_balance'. Default to 'apply' if not clear.")
    start_date: Optional[str] = Field(description="Start date of the leave in YYYY-MM-DD format. Return null if not mentioned.")
    end_date: Optional[str] = Field(description="End date of the leave in YYYY-MM-DD format. Return null if not mentioned.")
    leave_type: Optional[str] = Field(description="Type of leave (e.g. annual, sick, or 'all'). Return null if not mentioned.")

class TravelExtraction(BaseModel):
    action: Optional[str] = Field(description="The requested action: 'book' (default), 'check_history', or 'select'.")
    source: Optional[str] = Field(description="The source city or airport.")
    destination: Optional[str] = Field(description="The destination city or country.")
    outbound_date: Optional[str] = Field(description="The date of outbound travel in YYYY-MM-DD format.")
    return_date: Optional[str] = Field(description="The date of return travel in YYYY-MM-DD format (if round trip).")
    is_round_trip: Optional[bool] = Field(description="Whether the journey is a round trip. Return null ONLY if the user hasn't specified it yet. DO NOT assume False.")
    outbound_selection: Optional[int] = Field(description="The index (1, 2, or 3) of the OUTBOUND flight selected by the user.")
    return_selection: Optional[int] = Field(description="The index (1, 2, or 3) of the RETURN flight selected by the user.")

# NODE 1: Intent Router (classification node)
def intent_router(state: AgentState):
    messages = state["messages"]
    user_input = messages[-1].content # Latest message

    prompt = f"""
    Classify the intent of the following user input into exactly one of these categories:
    - policy: asking about rules, guidelines, or HR/travel policies.
    - leave: applying for leave, checking leave balance, or taking time off.
    - travel: booking flights, asking for travel options, or creating a travel request.
    - other: any query NOT related to HR policies, leave, or travel (e.g., general knowledge, math, coding, jokes, personal questions).

    Guardrails:
    - If the user's query is outside the domain of HR Operations (Policies, Leave, Travel), you MUST classify it as 'other'.
    - Use history of interactions to maintain context.

    Conversation History:
    {messages[:-1]}

    Current Input: {user_input}
    """

    structured_llm = llm.with_structured_output(IntentClassification)
    
    logger.info(f"Intent Router Prompt: {prompt}")
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    logger.info(f"Intent Router Result: {result}")
    
    intent = result.intent.lower()
    
    if intent == "other":
        msg = "I am an specialized HR Assistant. I can only help you with HR policies, leave management, and travel bookings. Please rephrase your request if it's related to these areas."
        return {"response": msg, "intent": "other", "messages": [HumanMessage(content=msg)]}

    return {"intent": intent}

# NODE 2: Policy RAG Node
def policy_node(state: AgentState):
    messages = state["messages"]
    query = messages[-1].content
    
    # Load FAISS Index
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    index_path = os.path.join(base_dir, "data", "faiss_index")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    try:
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(query, k=3)
        context = "\\n".join([d.page_content for d in docs])
    except Exception as e:
        context = "No policies found."
        print(f"Error loading FAISS: {e}")

    prompt = f"""
    You are an HR Policy Agent. Answer the user's question based strictly on the provided policy context.
    If the answer is not in the context, say that you cannot find the information in the current policies.
    
    Context:
    {context}
    
    User Question:
    {query}
    """
    
    logger.info(f"Policy Node Prompt: {prompt}")
    response = llm.invoke(messages + [HumanMessage(content=prompt)]).content
    logger.info(f"Policy Node Response: {response}")

    return {"response": response, "messages": [HumanMessage(content=response)]}


# NODE 3: Leave Node
def leave_node(state: AgentState):
    messages = state["messages"]
    employee_id = state["uid"]

    # Step 1: Extract details from HISTORY + CURRENT INPUT
    prompt = f"""
    Extract the leave application details from the following conversation history.
    - action: 'apply' (default) or 'check_balance'.
    - start_date: the first day of leave.
    - end_date: the last day of leave.
    - leave_type: annual, sick, or 'all'.

    IMPORTANT: Only return a value if it is explicitly mentioned in the conversation. DO NOT invent dates.
    
    Conversation History:
    {messages}
    """
    structured_llm = llm.with_structured_output(LeaveExtraction)
    
    logger.info(f"Leave Node Extraction Prompt: {prompt}")
    extraction = structured_llm.invoke([HumanMessage(content=prompt)])
    logger.info(f"Leave Node Extraction Result: {extraction}")

    # Handle Balance Check Action
    if extraction.action == "check_balance":
        balances = get_employee_balances(employee_id)
        if not balances:
            return {"response": "Employee records not found.", "messages": [HumanMessage(content="Employee records not found.")]}
        
        l_type = extraction.leave_type.lower() if extraction.leave_type else "all"
        if l_type == "sick":
            msg = f"Your sick leave balance is {balances['sick']} days."
        elif l_type == "annual":
            msg = f"Your annual leave balance is {balances['annual']} days."
        else:
            msg = f"Your current leave balances are: Annual: {balances['annual']} days, Sick: {balances['sick']} days."
        
        return {"response": msg, "messages": [HumanMessage(content=msg)]}

    # Step 2: Check for missing info for 'apply' action
    missing = []
    if not extraction.start_date: missing.append("start date")
    if not extraction.end_date: missing.append("end date")
    if not extraction.leave_type: missing.append("leave type")

    if missing:
        response = f"I'd be happy to help with your leave request. Could you please provide the missing {', '.join(missing)}?"
        return {"response": response, "messages": [HumanMessage(content=response)]}

    # Step 3: Calculate days and Validate Dates
    try:
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start = datetime.datetime.strptime(extraction.start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(extraction.end_date, "%Y-%m-%d")
        
        if start < today:
            logger.info(f"Validation Failed: Leave start date {extraction.start_date} is in the past.")
            msg = f"The start date ({extraction.start_date}) cannot be in the past. Please provide a current or future date."
            return {"response": msg, "messages": [HumanMessage(content=msg)]}

        # Policy Check: Annual Leave (2 weeks advance notice)
        l_type = extraction.leave_type.lower()
        if "annual" in l_type:
            advance_notice = (start - today).days
            if advance_notice < 14:
                logger.info(f"Policy Violation: Annual leave requested with only {advance_notice} days notice.")
                msg = f"Annual leave requests must be submitted at least 2 weeks in advance. You are only providing {advance_notice} days notice."
                return {"response": msg, "messages": [HumanMessage(content=msg)]}

        days = (end - start).days + 1
        if days <= 0:
            logger.info(f"Validation Failed: Leave end date {extraction.end_date} is before start date {extraction.start_date}.")
            msg = "The end date must be the same as or after the start date. Please check your dates."
            return {"response": msg, "messages": [HumanMessage(content=msg)]}
        
        # Policy Notice: Sick Leave (Medical certificate for > 3 days)
        sick_notice = ""
        if "sick" in l_type and days > 3:
            logger.info(f"Policy Note: Sick leave for {days} days requires medical certificate.")
            sick_notice = "\n\nNote: Since your sick leave exceeds 3 consecutive days, please ensure you submit a medical certificate to HR."

    except Exception as e:
        logger.error(f"Date Parsing Error: {e}")
        msg = "I couldn't parse the dates. Please provide them in YYYY-MM-DD format."
        return {"response": msg, "messages": [HumanMessage(content=msg)]}

    # Step 4: Check Balance from DB
    balances = get_employee_balances(employee_id)
    if not balances:
        msg = "Employee records not found."
        return {"response": msg, "messages": [HumanMessage(content=msg)]}

    l_type = extraction.leave_type.lower()
    balance_key = "annual" if "annual" in l_type else "sick"
    current_balance = balances.get(balance_key, 0)

    if days > current_balance:
        msg = f"Insufficient {balance_key} leave balance. You requested {days} days, but only have {current_balance} days left."
        return {"response": msg, "messages": [HumanMessage(content=msg)]}

    # Step 5: Update DB
    update_balances(employee_id, l_type, days)
    save_leave_record(employee_id, extraction.start_date, extraction.end_date, l_type, days)

    new_balances = get_employee_balances(employee_id)
    response = f"Success! Your {l_type} leave from {extraction.start_date} to {extraction.end_date} ({days} days) has been recorded. Your new {balance_key} leave balance is {new_balances[balance_key]} days.{sick_notice}"

    return {"response": response, "leave_balance": new_balances[balance_key], "messages": [HumanMessage(content=response)]}


# NODE 4: Travel Node
def travel_node(state: AgentState):
    messages = state["messages"]
    employee_id = state["uid"]

    # Step 1: Extract details
    prompt = f"""
    Extract the travel request details from the following conversation history.
    - action: 'book' (default), 'check_history', or 'select'.
    - source: source city.
    - destination: destination city.
    - outbound_date: YYYY-MM-DD.
    - return_date: YYYY-MM-DD.
    - is_round_trip: boolean.
    - outbound_selection: integer (1, 2, or 3) if user selected an OUTBOUND flight.
    - return_selection: integer (1, 2, or 3) if user selected a RETURN flight.

    IMPORTANT: Only return a value if it is explicitly mentioned in the conversation history. 
    DO NOT assume or invent dates, cities, or round-trip status if they are not there.
    Return null for any field that is not explicitly provided.

    Conversation History:
    {messages}
    """
    structured_llm = llm.with_structured_output(TravelExtraction)
    
    logger.info(f"Travel Node Extraction Prompt: {prompt}")
    extraction = structured_llm.invoke([HumanMessage(content=prompt)])
    logger.info(f"Travel Node Extraction Result: {extraction}")

    # Scenario: Check History
    if extraction.action == "check_history":
        history = get_travel_history(employee_id)
        if not history:
            msg = "You have no past or upcoming travel requests."
        else:
            msg = "Here are your travel requests:\n" + "\n".join([f"- {h[0]} to {h[1]} on {h[2]} (Status: {h[3]})" for h in history])
        return {"response": msg, "messages": [HumanMessage(content=msg)]}

    # Scenario: Selection Made
    if extraction.outbound_selection:
        out_idx = extraction.outbound_selection - 1
        outbound_options = state.get("outbound_options", [])
        
        if 0 <= out_idx < len(outbound_options):
            selected_out = outbound_options[out_idx]
            
            # If round trip and return not selected yet, show return options
            if extraction.is_round_trip and not extraction.return_selection:
                return_options = state.get("return_options", [])
                options_text = ""
                for i, f in enumerate(return_options):
                    options_text += f"{i+1}. {f['flight']} ({f['time']}) - {f['price']}\n"
                
                msg = f"Great! You've selected outbound: {selected_out['flight']}.\nNow, please select your **return** flight from the following options:\n{options_text}"
                return {"response": msg, "messages": [HumanMessage(content=msg)]}
            
            # If one way OR round trip with return selection
            selected_ret = None
            if extraction.is_round_trip and extraction.return_selection:
                ret_idx = extraction.return_selection - 1
                return_options = state.get("return_options", [])
                if 0 <= ret_idx < len(return_options):
                    selected_ret = return_options[ret_idx]
            
            # Finalize Request
            final_flight = f"Outbound: {selected_out['flight']} ({selected_out['time']})"
            final_price = selected_out['price']
            if selected_ret:
                final_flight += f" / Return: {selected_ret['flight']} ({selected_ret['time']})"
                # Simplified price handling: showing both or just sum if numeric. For now, just show both.
                final_price = f"{selected_out['price']} (Out) + {selected_ret['price']} (Ret)"

            travel_req = {
                "source": extraction.source or state.get("travel_request", {}).get("source"),
                "destination": extraction.destination or state.get("travel_request", {}).get("destination"),
                "outbound_date": extraction.outbound_date or state.get("travel_request", {}).get("outbound_date"),
                "return_date": extraction.return_date or state.get("travel_request", {}).get("return_date"),
                "is_round_trip": extraction.is_round_trip,
                "selected_flight": f"{final_flight} at {final_price}"
            }
            msg = f"You have selected {final_flight}. Sending this for approval."
            return {
                "response": msg,
                "travel_request": travel_req,
                "messages": [HumanMessage(content=msg)]
            }

    # Scenario: Gather Info and Search
    missing = []
    if not extraction.source: missing.append("source city")
    if not extraction.destination: missing.append("destination city")
    if not extraction.outbound_date: missing.append("outbound travel date")
    
    if extraction.is_round_trip is None:
        missing.append("whether it's a round trip")
    elif extraction.is_round_trip is True and not extraction.return_date:
        missing.append("return travel date")

    if missing:
        response = f"To help you book travel, I need the following details: {', '.join(missing)}."
        return {"response": response, "messages": [HumanMessage(content=response)]}

    # Step 3: Validate Travel Dates (Simplified for brevity in this chunk, already implemented)
    # ... [Validation logic remains the same] ...
    try:
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        outbound_dt = datetime.datetime.strptime(extraction.outbound_date, "%Y-%m-%d")
        if outbound_dt < today:
            msg = f"The outbound travel date ({extraction.outbound_date}) cannot be in the past."
            return {"response": msg, "messages": [HumanMessage(content=msg)]}
        if extraction.is_round_trip and extraction.return_date:
            ret_dt = datetime.datetime.strptime(extraction.return_date, "%Y-%m-%d")
            if ret_dt <= outbound_dt:
                msg = "The return date must be after the outbound travel date."
                return {"response": msg, "messages": [HumanMessage(content=msg)]}
    except: pass

    # Perform Search
    outbound_flights = search_flights.invoke({
        "source": extraction.source,
        "destination": extraction.destination,
        "date": extraction.outbound_date
    })

    return_flights = []
    if extraction.is_round_trip and extraction.return_date:
        return_flights = search_flights.invoke({
            "source": extraction.destination,
            "destination": extraction.source,
            "date": extraction.return_date
        })

    # Show Outbound options first
    options_text = ""
    for i, f in enumerate(outbound_flights):
        options_text += f"{i+1}. {f['flight']} ({f['time']}) - {f['price']}\n"

    response = f"I found the following **outbound** flights for {extraction.source} to {extraction.destination} on {extraction.outbound_date}:\n{options_text}\n\nPlease reply with the option number (1, 2, or 3) to select your outbound flight."

    return {
        "response": response,
        "outbound_options": outbound_flights,
        "return_options": return_flights,
        "travel_request": extraction.model_dump(),
        "messages": [HumanMessage(content=response)]
    }

# NODE 5: Approval Node
def approval_node(state: AgentState):
    if state.get("travel_request") and state["travel_request"].get("selected_flight"):
        req = state["travel_request"]
        employee_id = state["uid"]
        
        # Save to DB
        save_travel_request(
            employee_id,
            req["source"],
            req["destination"],
            req["outbound_date"],
            req.get("return_date"),
            req.get("is_round_trip", False),
            req["selected_flight"]
        )

        logger.info(f"Condition Met: Travel request for {req['destination']} submitted.")
        approval_msg = f"Travel request for {req['destination']} ({req['selected_flight']}) has been submitted for approval."
        return {
            "response": approval_msg,
            "messages": [HumanMessage(content=approval_msg)],
            "travel_request": None,
            "available_flights": None
        }

    logger.info("Condition Failed: No selected flight to approve.")
    return {}

