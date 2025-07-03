# Overview
This is a demo of a customer service agent of a telco company, built with LangChain/LangGraph, FastAPI and Streamlit.

The agent will be able to recommend mobile plans based on user's input and also reply with information about the terms of conditions of mobile and broadband plans.

## Quickstart
You need to provide your own OpenAI API key, else it will default to locally hosted LLM using Ollama. 

To run directly in Python:

```
# Install dependencies
uv sync --frozen

# Start FastAPI service
python run api_service.py

# In a different shell, launch Streamlit app
streamlit run streamlit_app.py
```

## Notes
- **Tools:**
  - `get_recommendation`: Recommends a list of mobile plans based on user input.
  - `lookup_tncs`: Retrieves information about terms and conditions for mobile and broadband plans.
- **Recommendation Workflow:**
  - Extracts relevant details from user input and standardizes them.
  - Calculates similarity between the request and available data plans.
  - Returns the closest matches to the agent. (Further fine-tuning can improve input parsing and matching.)
- **Terms & Conditions Lookup:**
  - Uses Retrieval-Augmented Generation (RAG) to fetch relevant documents.
  - Documents are split into parent and child chunks to balance context and relevancy.
  - Reranking can be considered to further boost retrieval performance.
- **API & Deployment:**
  - The API endpoint can be integrated with various platforms (e.g., Streamlit web app).
  - For production, consider adding authentication, checkpointing, guardrails, and batch API calls.
- **Performance & Feedback:**
  - Write and score test cases to assess agent performance.
  - In production, include user feedback mechanisms to monitor and improve the app.
