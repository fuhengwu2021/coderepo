"""
vLLM API Gateway - Automatically routes requests to corresponding vLLM Service based on 'model' field in request body
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import logging
from typing import Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM API Gateway")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model to Service mapping
# Format: {model_name_in_request: service_name}
MODEL_TO_SERVICE = {
    "meta-llama/Llama-3.2-1B-Instruct": "vllm-llama-32-1b",
    "/models/Phi-tiny-MoE-instruct": "vllm-phi-tiny-moe-service",
    "Phi-tiny-MoE-instruct": "vllm-phi-tiny-moe-service",  # 兼容不同的格式
}

# Service port (all Services use 8000)
SERVICE_PORT = 8000


def get_service_for_model(model: str) -> str:
    """Get corresponding Service name based on model name"""
    # Exact match
    if model in MODEL_TO_SERVICE:
        return MODEL_TO_SERVICE[model]
    
    # Fuzzy match (supports partial matching)
    for key, service in MODEL_TO_SERVICE.items():
        if key.lower() in model.lower() or model.lower() in key.lower():
            logger.info(f"Matched model '{model}' to service '{service}' via fuzzy match with '{key}'")
            return service
    
    # Return None if not found
    return None


async def forward_request(
    service_name: str,
    path: str,
    method: str,
    headers: Dict[str, str],
    body: bytes = None,
    params: Dict[str, Any] = None
) -> httpx.Response:
    """Forward request to corresponding Service"""
    # Build target URL
    target_url = f"http://{service_name}:{SERVICE_PORT}{path}"
    
    # Remove headers that shouldn't be forwarded
    forward_headers = {k: v for k, v in headers.items() 
                      if k.lower() not in ['host', 'content-length']}
    
    logger.info(f"Forwarding {method} {path} to {target_url}")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        if method == "GET":
            response = await client.get(target_url, headers=forward_headers, params=params)
        elif method == "POST":
            response = await client.post(
                target_url,
                headers=forward_headers,
                content=body,
                params=params
            )
        elif method == "DELETE":
            response = await client.delete(target_url, headers=forward_headers)
        else:
            raise HTTPException(status_code=405, detail=f"Method {method} not supported")
        
        return response


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "vllm-api-gateway"}


def normalize_model_id(model_id: str) -> str:
    """Normalize model ID (for deduplication)"""
    # Remove /models/ prefix if exists
    if model_id.startswith("/models/"):
        return model_id[8:]  # Remove "/models/" (8 characters)
    return model_id

@app.get("/v1/models")
async def list_models():
    """List all available models"""
    models = []
    seen_model_ids = set()  # For deduplication (store original IDs)
    seen_normalized_ids = set()  # For deduplication (store normalized IDs)
    
    # Collect all unique services (avoid calling same service multiple times)
    unique_services = set(MODEL_TO_SERVICE.values())
    
    for service_name in unique_services:
        try:
            # Try to get model information from corresponding Service
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://{service_name}:{SERVICE_PORT}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        # Only add unseen models (deduplicate based on normalized model id)
                        for model in data["data"]:
                            model_id = model.get("id", "")
                            if model_id:
                                normalized_id = normalize_model_id(model_id)
                                if normalized_id not in seen_normalized_ids:
                                    seen_model_ids.add(model_id)
                                    seen_normalized_ids.add(normalized_id)
                                    models.append(model)
        except Exception as e:
            logger.warning(f"Failed to get models from {service_name}: {e}")
    
    # If some services are unreachable, at least return model names from mapping (but deduplicate)
    for model_name in MODEL_TO_SERVICE.keys():
        normalized_name = normalize_model_id(model_name)
        if normalized_name not in seen_normalized_ids:
            seen_normalized_ids.add(normalized_name)
            models.append({
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "vllm"
            })
    
    return {"object": "list", "data": models}


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_v1(request: Request, path: str):
    """Proxy all /v1/* requests"""
    # Read request body
    body = await request.body()
    
    # Parse request body to get model field
    model = None
    if body:
        try:
            body_json = json.loads(body)
            model = body_json.get("model")
        except json.JSONDecodeError:
            pass
    
    # If no model field, try to get from query parameters
    if not model:
        model = request.query_params.get("model")
    
    # If still no model, return error
    if not model:
        raise HTTPException(
            status_code=400,
            detail="Missing 'model' field in request body or query parameters"
        )
    
    # Get corresponding Service
    service_name = get_service_for_model(model)
    if not service_name:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found. Available models: {list(MODEL_TO_SERVICE.keys())}"
        )
    
    logger.info(f"Routing request for model '{model}' to service '{service_name}'")
    
    # Forward request
    try:
        response = await forward_request(
            service_name=service_name,
            path=f"/v1/{path}",
            method=request.method,
            headers=dict(request.headers),
            body=body,
            params=dict(request.query_params)
        )
        
        # Handle streaming response
        if "text/event-stream" in response.headers.get("content-type", ""):
            async def generate():
                async for chunk in response.aiter_bytes():
                    yield chunk
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        # Return JSON response
        return JSONResponse(
            content=response.json() if response.content else {},
            status_code=response.status_code,
            headers=dict(response.headers)
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gateway timeout")
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Service '{service_name}' is not available. Please check if the pod is running."
        )
    except Exception as e:
        logger.error(f"Error forwarding request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_other(request: Request, path: str):
    """Proxy requests for other paths (e.g., /health, /metrics, etc.)"""
    # For non-/v1/* paths, try all Services (for health checks, etc.)
    # Or return 404
    if path in ["health", "metrics"]:
        # Try first available Service
        first_service = list(MODEL_TO_SERVICE.values())[0] if MODEL_TO_SERVICE else None
        if first_service:
            try:
                response = await forward_request(
                    service_name=first_service,
                    path=f"/{path}",
                    method=request.method,
                    headers=dict(request.headers),
                    body=await request.body(),
                    params=dict(request.query_params)
                )
                return JSONResponse(
                    content=response.json() if response.content else {},
                    status_code=response.status_code
                )
            except:
                pass
    
    raise HTTPException(status_code=404, detail=f"Path /{path} not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
