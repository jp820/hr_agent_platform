from fastapi import FastAPI
from pydantic import BaseModel
from hr_agent_platform.scripts.graphs import graph

app = FastAPI(title="HR Agent Platform API")

class QueryRequest(BaseModel):
    query: str
    uid: str
    tid: str

from langchain_core.messages import HumanMessage

def run_agent(user_query: str, uid: str, tid: str) -> str:
    # We use the thread ID (tid) for checkpointing so memory persists per conversation
    config = {"configurable": {"thread_id": tid}}
    
    # Initialize the input state with the new message and uid
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "uid": uid
    }
    
    # Invoke the graph
    result = graph.invoke(initial_state, config=config)
    
    return result.get("response", "I'm sorry, I could not process your request.")


# Define chat endpoint to interact with the agent
@app.post("/hr-chat")
def chat_endpoint(request: QueryRequest):
    user_query = request.query
    uid = request.uid # Get the unique user ID from the request
    tid = request.tid
    result = run_agent(user_query, uid, tid) # Run our agent with the user's query and ID
    return {"response": result}


'''
Sample curated_prompts API call:
curl --location 'http://localhost:8000/hr-chat' \
--header 'Content-Type: application/json' \
--data '{
    "user_profile": {
        "dietPreference": "Non-Vegetarian",
        "currentDiet": "High Protein",
        "dietGoal": "muscle gain",
        "taste": ["Italian pasta", "Neapolitan pizza", "Marathi meals"]
    },
    "personlised_search_credits": true,
    "timeStr": "09:35 AM",
    "day": "Friday"
}'

Sample hr-chat API call:
curl --location 'http://localhost:8000/hr-chat' \
--header 'Content-Type: application/json' \
--data '{
    "query": "What is leave policy?",
    "uid": "1",
    "tid": "1"
}'

curl --location 'http://localhost:8000/hr-chat' \
--header 'Content-Type: application/json' \
--data '{
    "query": "Apply leave from March 10 to March 15",
    "uid": "1",
    "tid": "1"
}'

curl --location 'http://localhost:8000/hr-chat' \
--header 'Content-Type: application/json' \
--data '{
    "query": "I need to travel to Bangalore next week",
    "uid": "1",
    "tid": "1"
}'
'''