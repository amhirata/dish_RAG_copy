# DishRAG Development Guide

## Commands
- **Run App**: `streamlit run app.py`
- **Install Dependencies**: `pip install -r requirements.txt`
- **Environment Setup**: Create `.env` file with API keys (openai_key, pinecone_key)

## Code Guidelines
- **Naming**: Use snake_case for variables and functions
- **Imports**: Group imports by type (core libraries, APIs, local)
- **Section Headers**: Use comment headers to separate code sections
- **Function Documentation**: Include docstrings for non-trivial functions
- **Error Handling**: Use conditional checks before operations
- **Parameters**: Use explicit parameter naming in function calls

## Project Structure
- **App Organization**: 
  1. Initial Setup (API connections)
  2. Helper Functions (RAG functionality)
  3. Streamlit App (UI components)
  4. Main Entry Point

## Security
- Store credentials in environment variables
- Use Streamlit secrets for deployed environments
- Firebase credentials should be stored securely

## Development Flow
- Test locally with `streamlit run app.py`
- Deployed to Streamlit Cloud (per commit history)