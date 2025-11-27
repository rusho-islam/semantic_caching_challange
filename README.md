# Semantic Caching Challenge

This project demonstrates a scalable, dockerized FastAPI-based service for large language model (LLM) inference using semantic caching with Redis. It integrates both local (Ollama) and OpenAI-hosted LLMs, seamlessly switching between them as required.

---

## Features

- **FastAPI microservice** for handling LLM inference requests.
- **Semantic caching** using Redis to avoid redundant LLM calls for semantically similar queries, reducing LLM usage costs and response times.
- **Pluggable LLM backends**: Use OpenAI models or local [Ollama](https://github.com/ollama/ollama) models.
- **Vector similarity search** for semantic lookup of prompts (using models from `sentence-transformers`).
- **Dockerized**: Easily run locally, on-premises, or in the cloud.
- **Environment variables** managed by `.env` and `python-dotenv`.
- **Reproducible builds** using `uv` and lock files.

---

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose installed
- (Optional) [Ollama](https://github.com/ollama/ollama) model files for local LLM
- OpenAI API Key, if you want to use OpenAI models

### 2. Clone the Repository

```sh
git clone https://github.com/your_org/semantic_caching_challange.git
cd semantic_caching_challange
```

### 3. Set Up Environment Variables

Copy `.env.example` to `.env` and fill in your OpenAI key and other variables as needed.

### 4. Build & Run the Stack

```sh
docker-compose up --build
```

- The FastAPI server will be available at: `http://localhost:3000`
- Ollama server at: `http://localhost:11434`
- Redis at: `localhost:6379`

---

## Usage

### API Endpoint

- `POST /query`
    - Body: `{ "question": "your prompt here" }`
    - Returns: The LLM's response (from cache or fresh generation)

### Example Request

```bash
curl --location --request GET 'http://localhost:3000/query/' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "query": "whats the capital of farnce?",
        "forceRefresh": true
    }'
```

---

## Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── models/
│   └── llm_model.py
├── service.py
├── utils/
│   └── utils.py
├── config/
│   └── config.py
├── README.md
└── ...
```

---

## Technology Stack

- **FastAPI** for API layer
- **transformers** & **sentence-transformers** for embeddings and scoring semantic similarity
- **Redis** for caching vector embeddings and responses
- **LangChain** for LLM integration
- **Ollama** & **OpenAI** as language model providers
- **Docker** for reproducible environments

---

## Development & Testing

To run/tests locally:

```sh
# Install dependencies (use Python 3.11+)
uv pip install -r requirements.txt

# Run FastAPI server
uvicorn service:app --reload --host 0.0.0.0 --port 3000
```

---

## Acknowledgments

- [Ollama](https://github.com/ollama/ollama)
- [LangChain](https://python.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Redis](https://redis.io/)

---

## License

[MIT License](./LICENSE)
 
---

