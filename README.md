# 🐍 Universal Agent RAG - Backend API

A high-performance FastAPI backend service powered by Google Gemini 2.5 Flash for intelligent conversation generation and RAG (Retrieval-Augmented Generation) capabilities.

## 🚀 Overview

This backend service provides a RESTful API for AI-powered conversations using Google's Gemini 2.5 Flash model through Langchain integration. It features comprehensive logging, request tracking, and optimized performance for production deployments.

## 🏗️ Architecture

```
app-backend/
├── src/
│   ├── server.py              # Main FastAPI application
│   └── rag/
│       └── main.py            # LLM configuration and setup
├── pyproject.toml             # Python dependencies (uv)
├── uv.lock                    # Locked dependencies
├── Dockerfile                 # Container image
├── .env                       # Environment variables
└── README.md                  # This file
```

## 🛠️ Technology Stack

- **FastAPI** - Modern, fast web framework for building APIs
- **Python 3.12+** - Latest Python version with performance improvements
- **Google Gemini 2.5 Flash** - Advanced LLM for conversation generation
- **Langchain** - LLM integration and orchestration framework
- **Pydantic** - Data validation and serialization
- **UV** - Ultra-fast Python package manager
- **Uvicorn** - Lightning-fast ASGI server

## ⚡ Features

### ✅ Current Features
- 🤖 **AI Chat Generation** with Google Gemini 2.5 Flash
- 📊 **Comprehensive Logging** with request tracking
- 🔍 **Request/Response Monitoring** with performance metrics
- 🛡️ **Input Validation** using Pydantic models
- 🚀 **High Performance** with async/await patterns
- 🐳 **Docker Ready** with optimized container builds
- 📈 **Production Logging** to files and console
- 🔧 **Health Check** endpoints

### 🚧 Planned Features
- 📡 **Streaming Responses** (Server-Sent Events)
- 🗄️ **Database Integration** for conversation persistence
- 🔍 **RAG Implementation** with vector embeddings
- 🔐 **Authentication & Authorization**
- 📊 **Metrics & Analytics** endpoints
- 🧪 **Unit & Integration Tests**

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ (for local development)
- UV package manager (recommended) or pip
- Google API key for Gemini access

### 🐳 Docker Deployment (Recommended)

```bash
# Build and run with Docker
docker build -t universal-agent-backend .
docker run -p 8080:8080 --env-file .env universal-agent-backend
```

### 🛠️ Local Development

1. **Clone and navigate:**
   ```bash
   cd app-backend
   ```

2. **Install UV (if not installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the server:**
   ```bash
   uv run src/server.py
   ```

### 📦 Traditional pip Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi[standard] langchain-google-genai python-dotenv

# Run server
python src/server.py
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the app-backend directory:

```env
# Required API Keys
GOOGLE_API_KEY=your_google_gemini_api_key

# Optional: Satellite API for additional features
N2YO_API_KEY=your_n2yo_satellite_api_key

# Application Settings
APP_ENV=development
APP_DEBUG=true
PORT=8080

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=app.log
```

### LLM Configuration

The Gemini model configuration in `src/rag/main.py`:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",    # Model version
    temperature=0,               # Deterministic responses
    max_tokens=None,             # No token limit
    timeout=None,                # No timeout limit
    max_retries=2,               # Retry failed requests
)
```

## 📡 API Endpoints

### Base URL
- **Local**: `http://localhost:8080`
- **Docker**: `http://localhost:8080`
- **Production**: `http://your-domain:8080`

### Available Endpoints

#### 🏠 Health Check
```http
GET /
```

**Response:**
```json
{
  "Hello": "World"
}
```

#### 🤖 Generate AI Response
```http
POST /generation
Content-Type: application/json
```

**Request Body:**
```json
{
  "prompt": "Hello, how can you help me today?"
}
```

**Response:**
```json
{
  "model": "gemini-2.5-flash",
  "generation_id": "gen-550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1727347200.123,
  "response": "Hello! I'm here to help you with any questions or tasks you might have..."
}
```

**Response Fields:**
- `model`: The LLM model used for generation
- `generation_id`: Unique identifier for this generation
- `timestamp`: Unix timestamp of the response
- `response`: The AI-generated response text

## 🔍 Logging & Monitoring

### Logging Features

The backend includes comprehensive logging with:

- **Request Tracking**: Each request gets a unique ID
- **Performance Metrics**: Response time measurement
- **Client Information**: IP, User-Agent tracking
- **Request/Response Logging**: Payload logging (truncated for privacy)
- **Error Handling**: Detailed exception logging
- **File & Console Output**: Dual logging destinations

### Log Format

```
2025-09-26 10:30:45,123 - INFO - [req-123] Request received | POST /generation | Client: 172.17.0.1 | UA: curl/7.68.0...
2025-09-26 10:30:45,456 - INFO - [req-123] Request payload: {"prompt": "Hello world"}
2025-09-26 10:30:46,789 - INFO - [req-123] Response | Status: 200 | Time: 1.234s | Size: 256 bytes
```

### Log Files

- **Location**: `app.log` in the application directory
- **Rotation**: Manual rotation recommended for production
- **Levels**: INFO, WARNING, ERROR with appropriate filtering

## 🐳 Docker Configuration

### Dockerfile Optimization

The Dockerfile uses multi-stage optimization:

1. **Base Image**: `python:3.12-slim-bookworm` for minimal size
2. **UV Integration**: Ultra-fast dependency management
3. **Layer Caching**: Optimized build order for faster rebuilds
4. **Security**: Non-root user execution (planned)

### Build Arguments

```bash
# Custom build with different Python version
docker build --build-arg PYTHON_VERSION=3.11 -t backend .

# Development build with additional tools
docker build --target development -t backend-dev .
```

## 🧪 Testing

### Manual Testing

```bash
# Health check
curl http://localhost:8080/

# AI generation
curl -X POST http://localhost:8080/generation \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is artificial intelligence?"}'
```

### Load Testing

```bash
# Using Apache Bench
ab -n 100 -c 10 -T application/json -p prompt.json http://localhost:8080/generation

# Using wrk
wrk -t12 -c400 -d30s --script=post.lua http://localhost:8080/generation
```

## 🚀 Production Deployment

### Performance Tuning

1. **Worker Processes:**
   ```bash
   uvicorn src.server:app --workers 4 --host 0.0.0.0 --port 8080
   ```

2. **Resource Limits:**
   ```yaml
   # docker-compose.yml
   services:
     app:
       deploy:
         resources:
           limits:
             memory: 1G
             cpus: "0.5"
   ```

### Security Considerations

- **API Keys**: Store in secure environment variables
- **CORS**: Configure allowed origins in production
- **Rate Limiting**: Implement request rate limiting
- **Input Sanitization**: Validate and sanitize all inputs
- **Logging**: Avoid logging sensitive information

### Monitoring

- **Health Checks**: Implement liveness/readiness probes
- **Metrics**: Add Prometheus metrics endpoints
- **Alerts**: Set up monitoring alerts for errors/latency
- **Log Aggregation**: Use centralized logging solutions

## 🔧 Development

### Code Structure

```python
# server.py - Main application structure
app = FastAPI(root_path="/api/v1")  # API versioning

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str

# Middleware for logging
@app.middleware("http")
async def log_requests(request, call_next):
    # Request tracking and performance monitoring
```

### Adding New Endpoints

1. **Define Request Model:**
   ```python
   class NewRequest(BaseModel):
       field: str
   ```

2. **Create Endpoint:**
   ```python
   @app.post("/new-endpoint")
   async def new_endpoint(request: NewRequest):
       # Implementation
       return {"result": "success"}
   ```

3. **Add Logging:**
   ```python
   logging.info(f"Processing new endpoint: {request.field}")
   ```

### Debugging

```bash
# Enable debug mode
export APP_DEBUG=true

# Increase log verbosity
export LOG_LEVEL=DEBUG

# Run with hot reload
uvicorn src.server:app --reload --log-level debug
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Follow** code style guidelines:
   - Use **type hints** for all functions
   - Follow **PEP 8** formatting
   - Add **docstrings** for public functions
   - Include **error handling**
4. **Test** your changes thoroughly
5. **Commit** with descriptive messages
6. **Push** and create a Pull Request

### Code Style

```python
from typing import Optional, Dict, Any
import logging

async def example_function(
    request: GenerationRequest,
    optional_param: Optional[str] = None
) -> Dict[str, Any]:
    """
    Example function with proper typing and documentation.
    
    Args:
        request: The generation request model
        optional_param: Optional parameter for customization
        
    Returns:
        Dictionary containing the response data
        
    Raises:
        ValueError: If request validation fails
    """
    try:
        # Implementation here
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Error in example_function: {str(e)}")
        raise
```

## 📄 License

This project is licensed under the MIT License. See the main project LICENSE file for details.

## 🔗 Related

- **Frontend**: `../web-frontend/README.md`
- **Main Project**: `../README.md`
- **Docker Compose**: `../docker-compose.yml`
- **NGINX Config**: `../nginx.conf`

## 🙏 Acknowledgments

- **Google Gemini** for providing the LLM capabilities
- **Langchain** for LLM integration framework
- **FastAPI** for the excellent web framework
- **UV** for ultra-fast package management

---

*Built with ❤️ for high-performance AI applications.*
