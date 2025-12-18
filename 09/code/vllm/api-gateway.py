"""
LLM API Gateway - Automatically routes requests to corresponding Service based on 'model' and 'inference_server'/'engine' fields in request body
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import logging
from typing import Dict, Any, Optional, Tuple, List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM API Gateway")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routing configuration (mutable for dynamic updates)
# Format: {(model_name, inference_server): service_name}
# inference_server can be "vllm", "sglang", etc.
# If inference_server is None, it matches any inference server
ROUTING_CONFIG: Dict[Tuple[str, Optional[str]], str] = {
    # vLLM services
    ("meta-llama/Llama-3.2-1B-Instruct", "vllm"): "vllm-llama-32-1b",
    ("meta-llama/Llama-3.2-1B-Instruct", None): "vllm-llama-32-1b",  # Default to vLLM if not specified
    
    # SGLang services
    ("meta-llama/Llama-3.2-1B-Instruct", "sglang"): "sglang-llama-32-1b",
}

# Pydantic models for API
class RoutingMapping(BaseModel):
    model: str
    inference_server: Optional[str] = None  # None means default/any
    service_name: str

class RoutingMappingResponse(BaseModel):
    model: str
    inference_server: Optional[str]
    service_name: str

# Service port (all Services use 8000)
SERVICE_PORT = 8000


def get_service_for_model_and_engine(model: str, inference_server: Optional[str] = None) -> Optional[str]:
    """
    Get corresponding Service name based on model name and inference_server/engine
    
    Args:
        model: Model name from request
        inference_server: Inference server type (vllm, sglang, etc.) from 'inference_server' or 'engine' field
    
    Returns:
        Service name or None if not found
    """
    # Normalize inference_server
    if inference_server:
        inference_server = inference_server.lower()
    
    # Try exact match with inference_server
    if inference_server:
        key = (model, inference_server)
        if key in ROUTING_CONFIG:
            service = ROUTING_CONFIG[key]
            if service:
                logger.info(f"Matched model '{model}' with inference_server '{inference_server}' to service '{service}'")
                return service
    
    # Try match with None (default/any inference_server)
    key = (model, None)
    if key in ROUTING_CONFIG:
        service = ROUTING_CONFIG[key]
        if service:
            logger.info(f"Matched model '{model}' (default inference_server) to service '{service}'")
            return service
    
    # Fuzzy match - try to find similar model names
    model_lower = model.lower()
    for (key_model, key_server), service in ROUTING_CONFIG.items():
        if service is None:
            continue
        
        # Check if model matches (fuzzy)
        key_model_lower = key_model.lower() if key_model else ""
        if key_model_lower in model_lower or model_lower in key_model_lower:
            # If inference_server matches or key_server is None (default)
            if (key_server is None) or (inference_server and key_server == inference_server.lower()):
                logger.info(f"Matched model '{model}' (fuzzy) with inference_server '{inference_server}' to service '{service}'")
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
    return {"status": "healthy", "service": "llm-api-gateway"}


# ============================================================================
# Routing Configuration Management API
# ============================================================================

@app.get("/admin/api/routing", response_model=List[RoutingMappingResponse])
async def get_routing_config():
    """Get all routing mappings"""
    mappings = []
    for (model, inference_server), service_name in ROUTING_CONFIG.items():
        mappings.append(RoutingMappingResponse(
            model=model,
            inference_server=inference_server,
            service_name=service_name
        ))
    return mappings


@app.post("/admin/api/routing", response_model=RoutingMappingResponse)
async def add_routing_mapping(mapping: RoutingMapping):
    """Add a new routing mapping"""
    key = (mapping.model, mapping.inference_server)
    if key in ROUTING_CONFIG:
        raise HTTPException(status_code=400, detail=f"Mapping already exists for model '{mapping.model}' with inference_server '{mapping.inference_server}'")
    
    ROUTING_CONFIG[key] = mapping.service_name
    logger.info(f"Added routing mapping: {key} -> {mapping.service_name}")
    
    return RoutingMappingResponse(
        model=mapping.model,
        inference_server=mapping.inference_server,
        service_name=mapping.service_name
    )


@app.delete("/admin/api/routing")
async def delete_routing_mapping(model: str, inference_server: Optional[str] = None):
    """Delete a routing mapping"""
    # Convert empty string to None
    if inference_server == "":
        inference_server = None
    
    key = (model, inference_server)
    if key not in ROUTING_CONFIG:
        raise HTTPException(status_code=404, detail=f"Mapping not found for model '{model}' with inference_server '{inference_server}'")
    
    del ROUTING_CONFIG[key]
    logger.info(f"Deleted routing mapping: {key}")
    
    return {"message": "Mapping deleted successfully"}


def normalize_model_id(model_id: str) -> str:
    """Normalize model ID (for deduplication)"""
    # Remove /models/ prefix if exists
    if model_id.startswith("/models/"):
        return model_id[8:]  # Remove "/models/" (8 characters)
    return model_id

@app.get("/v1/models")
async def list_models():
    """List all available models from all inference services
    
    Each (model, engine) combination is shown as a separate model entry.
    """
    models = []
    seen_keys = set()  # Track (model_id, inference_server) combinations
    
    # Build mapping: service_name -> list of (model_name, inference_server) tuples
    # But filter out None (default) mappings if there's already a specific engine for the same model+service
    service_to_mappings = {}
    model_service_has_specific_engine = {}  # Track if a model+service has specific engines
    
    # First pass: collect all mappings and track which have specific engines
    for (model_name, inference_server), service_name in ROUTING_CONFIG.items():
        if service_name and service_name is not None:
            key = (model_name, service_name)
            if inference_server is not None:
                model_service_has_specific_engine[key] = True
            elif key not in model_service_has_specific_engine:
                model_service_has_specific_engine[key] = False
    
    # Second pass: build mappings, skipping None if specific engine exists
    for (model_name, inference_server), service_name in ROUTING_CONFIG.items():
        if service_name and service_name is not None:
            key = (model_name, service_name)
            # Skip None (default) mapping if there's already a specific engine for this model+service
            if inference_server is None and model_service_has_specific_engine.get(key, False):
                continue  # Skip default mapping since we have specific engines
            
            if service_name not in service_to_mappings:
                service_to_mappings[service_name] = []
            service_to_mappings[service_name].append((model_name, inference_server))
    
    # For each service, get models and create entries for each (model, engine) combination
    for service_name, mappings in service_to_mappings.items():
        try:
            # Get models from the service
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://{service_name}:{SERVICE_PORT}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        # For each model returned by the service
                        for model in data["data"]:
                            model_id = model.get("id", "")
                            if model_id:
                                # Find matching mappings for this model
                                for model_name, inference_server in mappings:
                                    # Check if this model matches the mapping
                                    # Use fuzzy matching: model_id should match model_name
                                    if model_id == model_name or model_name in model_id or model_id in model_name:
                                        # Create unique key: (model_id, inference_server)
                                        key = (model_id, inference_server)
                                        if key not in seen_keys:
                                            seen_keys.add(key)
                                            engine_name = inference_server or "default"
                                            
                                            # Create a separate model entry for each engine
                                            model_entry = model.copy()
                                            model_entry["id"] = model_id  # Clean base model ID
                                            # Remove permission field if present (it's vLLM-specific)
                                            model_entry.pop("permission", None)
                                            # Remove engine field - owned_by already indicates the engine
                                            model_entry.pop("engine", None)
                                            # Ensure owned_by is set to the engine name
                                            model_entry["owned_by"] = engine_name
                                            models.append(model_entry)
        except Exception as e:
            logger.warning(f"Failed to get models from {service_name}: {e}")
    
    # If no models from services, add fallback from routing config
    if not models:
        for (model_name, inference_server), service_name in ROUTING_CONFIG.items():
            if model_name and service_name is not None:
                key = (model_name, inference_server)
                if key not in seen_keys:
                    seen_keys.add(key)
                    engine_name = inference_server or "default"
                    
                    models.append({
                        "id": model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": engine_name  # Use engine name as owned_by
                    })
    
    return models


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_v1(request: Request, path: str):
    """Proxy all /v1/* requests"""
    # Read request body
    body = await request.body()
    
    # Parse request body to get model and inference_server/engine/owned_by fields
    model = None
    inference_server = None
    if body:
        try:
            body_json = json.loads(body)
            model = body_json.get("model")
            # Check for inference_server, engine, or owned_by field (owned_by can be used for routing)
            inference_server = (body_json.get("inference_server") or 
                              body_json.get("engine") or 
                              body_json.get("owned_by"))
        except json.JSONDecodeError:
            pass
    
    # If no model field, try to get from query parameters
    if not model:
        model = request.query_params.get("model")
    
    # If no inference_server, try to get from query parameters (including owned_by)
    if not inference_server:
        inference_server = (request.query_params.get("inference_server") or 
                          request.query_params.get("engine") or 
                          request.query_params.get("owned_by"))
    
    # If still no model, return error
    if not model:
        raise HTTPException(
            status_code=400,
            detail="Missing 'model' field in request body or query parameters"
        )
    
    # Handle model IDs with engine suffix (format: model_id::engine::engine_name)
    original_model = model
    if "::engine::" in model:
        parts = model.split("::engine::")
        if len(parts) == 2:
            actual_model = parts[0]
            model_engine = parts[1] if parts[1] != "default" else None
            # Use the engine from model ID if inference_server not specified
            if not inference_server:
                inference_server = model_engine
            model = actual_model
    
    # Get corresponding Service based on model and inference_server
    service_name = get_service_for_model_and_engine(model, inference_server)
    if not service_name:
        available_models = [f"{m} (inference_server: {s})" if s else m 
                           for (m, s) in ROUTING_CONFIG.keys() if m]
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' with inference_server '{inference_server}' not found. "
                   f"Available: {available_models}"
        )
    
    logger.info(f"Routing request for model '{model}' with inference_server '{inference_server}' to service '{service_name}'")
    
    # Forward request - replace model ID in body if it had engine suffix
    forward_body = body
    if body and "::engine::" in original_model:
        try:
            body_json = json.loads(body)
            if "model" in body_json:
                body_json["model"] = model  # Replace with actual model name
                forward_body = json.dumps(body_json).encode('utf-8')
        except json.JSONDecodeError:
            pass  # If body parsing fails, use original body
    
    # Forward request
    try:
        response = await forward_request(
            service_name=service_name,
            path=f"/v1/{path}",
            method=request.method,
            headers=dict(request.headers),
            body=forward_body,
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
        unique_services = {service for service in ROUTING_CONFIG.values() if service is not None}
        first_service = list(unique_services)[0] if unique_services else None
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
