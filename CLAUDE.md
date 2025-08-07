# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the application
- **Start API server**: `python main.py` or use `llm_api_start.bat` (includes conda environment activation)
- **Install dependencies**: Use `llm_install.bat` or manually install from `requirements1.txt`, `requirements2.txt`, `requirements3.txt`
- **Port**: Default 5000 (configurable via TUNNEL_PORT environment variable)

### Testing
- **API endpoint testing**: Use FastAPI's built-in docs at `http://localhost:5000/docs`
- **No automated tests found** - Consider implementing unit tests for modules

## Architecture

### Core Components

1. **FastAPI Application** (`main.py`)
   - Main entry point for the RAG (Retrieval-Augmented Generation) system
   - Provides REST API endpoints for chat, vector database management, and model training
   - Integrates with both OpenAI models and local Unsloth/Gemma models

2. **Vector Database System**
   - Uses ChromaDB for vector storage
   - Supports hybrid search combining BM25 (keyword) and semantic search
   - Stores embeddings using HuggingFace multilingual models
   - Database files stored in `vector_dbs/` directory

3. **Model Management**
   - Supports both cloud models (OpenAI GPT-4, GPT-4o-mini) and local models (Unsloth/Gemma variants)
   - LoRA fine-tuning capabilities for local models
   - Dynamic model loading/unloading to manage GPU memory
   - Trained models stored in `trained_models/` directory

4. **Configuration** (`config.ini`)
   - Central configuration for API keys, model paths, and default parameters
   - Separate sections for EMBED_API_DEFAULTS and CHAT_API_DEFAULTS

### Key API Endpoints

- `/api/embed` - Create vector databases from uploaded documents
- `/api/chat` - Chat with RAG support and optional image input (OpenAI models only)
- `/api/train` - Fine-tune models using LoRA
- `/api/model/list` - List available models
- `/api/embed/list` - List vector databases
- `/api/data/augment` - Data augmentation for Q&A pairs

### Module Structure

- `modules/config.py` - Configuration management
- `modules/file_processing.py` - Document processing (PDF, DOCX, TXT)
- `modules/vector_db.py` - Vector database operations
- `modules/chat_history.py` - User chat history management
- `modules/llm_caller_unsloth.py` - Local model inference
- `utils/` - Utility functions for data processing

### External Dependencies

- Requires conda environment `MIRDC_Unsloth_clone` with Python 3.10
- GPU support required for local model inference
- Tunnel setup available via `tunnel_setup.py` for external access