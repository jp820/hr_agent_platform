# HR Intelligent Assistant

A production-grade, stateful HR Chatbot platform built with **LangGraph**, **Gemini**, and **SQLite**. This agent helps employees manage HR policies, leave applications, and travel bookings through an interactive, multi-turn conversational interface.

## Key Features

- **HR Policy RAG**: Query company policies (Leave, Travel, Remote Work, etc.) using a FAISS-backed vector database.
- **Intelligent Leave Management**: 
    - Check real-time leave balances (Annual, Sick).
    - Apply for leave with automatic policy validation (e.g., 2-week notice for annual leave).
- **Interactive Travel Booking**:
    - Multi-turn gathering of travel details (source, destination, dates).
    - **Real-time Flight Search**: Uses Google Search grounding via Gemini to find actual flights.
    - Support for round-trip planning and historical travel logs.
- **Stateful Memory**: Maintains context across multiple turns using LangGraph checkpoints and SQLite.

## Architecture

The agent follows a modular graph-based architecture using **LangGraph**:

1. **Intent Router**: The entry point that classifies user input into `policy`, `leave`, or `travel`.
2. **Policy Node (RAG)**: Performs a similarity search on `data/policies.txt` to answer HR questions.
3. **Leave Node**: Validates leave balances and HR rules (e.g., medical certificates for 3+ days of sick leave) before updating the SQLite DB.
4. **Travel Node**: Handles the multi-turn extraction of travel parameters and triggers real-time flight searches.
5. **Approval Node**: Persists confirmed travel/leave requests into the database.
6. **Persistence Layer**: 
    - **SQLite**: Stores employee data and request history.
    - **LangGraph Checkpointer**: Saves conversation state per `thread_id`.

## Usage Examples

### 1. HR Policy Inquiry
> **User**: "What is the policy for annual leave?"
> **Agent**: "Employees are entitled to 20 days of paid annual leave per year. Requests must be submitted at least 2 weeks in advance."

### 2. Applying for Leave
> **User**: "Apply annual leave from 2026-11-10 to 2026-11-15"
> **Agent**: "Success! Your annual leave has been recorded. Your new balance is 15 days."

### 3. Travel Booking (Multi-turn)
> **User**: "I need to book a flight to Kolkata."
> **Agent**: "I need the source city, travel date, and if it's a round trip."
> **User**: "From Mumbai on 2026-12-01. One way."
> **Agent**: [Lists real-time flight options from Google Search]
> **User**: "Option 1 please."
> **Agent**: "Travel request for Kolkata has been submitted for approval."

## 💻 Getting Started

1. **Environment Setup**:
   ```bash
   conda create -n proj_env python=3.11
   conda activate proj_env
   pip install -r requirements.txt
   export GOOGLE_API_KEY="api_key"
   ```

2. **Initialize Database**:
   ```bash
   python src/hr_agent_platform/scripts/init_db.py
   ```

3. **Run Streamlit Demo**:
   ```bash
   streamlit run src/hr_agent_platform/app/streamlit_app.py
   ```

## TO DO
1. Leave request history
2. Weekend & Holiday exclusion while calculating leave days
3. Role based authentication. HR, Admin and Employee
4. Summarization of past conversations to increase LLM efficiency
5. Human in loop for approval of leave/travel requests
6. Proper error handling and logging

---
