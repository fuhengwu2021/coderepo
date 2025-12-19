"""
llm-d API Gateway - Routes requests to ModelService based on 'model' and 'owned_by' fields

This gateway supports routing by both model name and inference engine type (owned_by).
It can read 'owned_by' from request body or HTTP header (x-owned-by).
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import logging
import os
from typing import Dict, Any, Optional, Tuple, List
from kubernetes import client, config
from kubernetes.client.rest import ApiException
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="llm-d API Gateway")
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Service port (all ModelServices use 8000)
SERVICE_PORT = 8000
NAMESPACE = os.getenv("NAMESPACE", "llm-d-multi-model")
# Initialize Kubernetes client
try:
    config.load_incluster_config()
    logger.info("Loaded in-cluster Kubernetes config")
except:
    try:
        config.load_kube_config()
        logger.info("Loaded kubeconfig from default location")
    except Exception as e:
        logger.warning(f"Could not load Kubernetes config: {e}. Service discovery will be limited.")
v1 = client.CoreV1Api()
# Routing configuration (can be updated dynamically)
# Format: {(model_name, owned_by): service_name}
# owned_by can be "vllm", "sglang", etc., or None for default
# This will be auto-populated from Kubernetes Services, but can be manually configured
ROUTING_CONFIG: Dict[Tuple[str, Optional[str]], str] = {
    # Example: Add your models here or let auto-discovery populate it
    # ("meta-llama/Llama-3.2-1B-Instruct", "vllm"): "ms-llama-32-1b-llm-d-modelservice-decode",
    # ("Qwen/Qwen2.5-0.5B-Instruct", "vllm"): "ms-qwen2-5-0-5b-llm-d-modelservice-decode",
}
async def discover_services():
    """Discover ModelService instances from Kubernetes"""
    try:
        import asyncio
        
        # Discover from pods (more reliable for getting actual model names)
        pods = v1.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector="llm-d.ai/role=decode"
        )
        
        discovered = {}
        pod_to_service = {}  # Map pod IP to service name
        
        # First, get all services to map pod IPs to service names
        services = v1.list_namespaced_service(namespace=NAMESPACE)
        for svc in services.items:
            endpoints = v1.read_namespaced_endpoints(svc.metadata.name, NAMESPACE)
            for subset in endpoints.subsets or []:
                for address in subset.addresses or []:
                    if address.target_ref and address.target_ref.kind == "Pod":
                        pod_to_service[address.ip] = svc.metadata.name
        
        async def query_single_pod(pod, pod_ip, pod_to_service):
            """Query a single pod to get its model name"""
            service_name = pod_to_service.get(pod_ip, "")
            model_name = None
            
            # Try to query the pod directly to get actual model name
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    # Query pod directly via IP
                    response = await client.get(f"http://{pod_ip}:{SERVICE_PORT}/v1/models")
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and len(data["data"]) > 0:
                            # Get the first model's ID (each pod typically serves one model)
                            model_name = data["data"][0].get("id", "")
                            logger.info(f"Queried pod {pod.metadata.name} ({pod_ip}): found model '{model_name}'")
            except Exception as e:
                logger.warning(f"Failed to query pod {pod.metadata.name} ({pod_ip}): {e}")
            
            # Fallback: try to extract from pod name
            if not model_name:
                pod_name = pod.metadata.name
                # ms-llama-32-1b-llm-d-modelservice-decode-xxx -> llama-32-1b
                if "ms-" in pod_name:
                    parts = pod_name.replace("ms-", "").split("-llm-d-modelservice-decode")[0]
                    # Convert to model path format
                    # llama-32-1b -> meta-llama/Llama-3.2-1B-Instruct (approximate)
                    # qwen2-5-0-5b -> Qwen/Qwen2.5-0.5B-Instruct (approximate)
                    if "llama" in parts.lower():
                        model_name = "meta-llama/Llama-3.2-1B-Instruct"
                    elif "qwen" in parts.lower():
                        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            
            # Default to vllm if no owned_by specified
            labels = pod.metadata.labels or {}
            owned_by = labels.get("llm-d.ai/owned-by", "vllm")
            if not owned_by or owned_by == "":
                owned_by = "vllm"
            
            # Return discovered mapping
            result = {}
            if model_name:
                if service_name:
                    key = (model_name, owned_by)
                    result[key] = service_name
                    logger.info(f"Discovered: {model_name} (owned_by: {owned_by}) -> {service_name} (pod: {pod.metadata.name})")
                else:
                    # Fallback: use pod IP directly
                    key = (model_name, owned_by)
                    result[key] = pod_ip
                    logger.info(f"Discovered: {model_name} (owned_by: {owned_by}) -> pod IP {pod_ip} (no service found)")
            return result
        
        # Query all pods in parallel
        tasks = []
        for pod in pods.items:
            pod_ip = pod.status.pod_ip
            if not pod_ip:
                continue
            tasks.append(query_single_pod(pod, pod_ip, pod_to_service))
        
        # Run all queries in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, dict) and result:
                    discovered.update(result)
        
        # Update routing config with discovered services
        ROUTING_CONFIG.update(discovered)
        return discovered
    except Exception as e:
        logger.warning(f"Failed to discover services from Kubernetes: {e}")
        return {}
# Discover services on startup (will be called asynchronously)
# Note: discover_services is now async, so we'll call it on first request or via admin API
# Pydantic models
class RoutingMapping(BaseModel):
    model: str
    owned_by: Optional[str] = None
    service_name: str
class RoutingMappingResponse(BaseModel):
    model: str
    owned_by: Optional[str]
    service_name: str
def get_service_for_model_and_owned_by(model: str, owned_by: Optional[str] = None) -> Optional[str]:
    """
    Get corresponding Service name based on model name and owned_by
    
    Args:
        model: Model name from request
        owned_by: Inference engine type (vllm, sglang, etc.) from 'owned_by' field or header
    
    Returns:
        Service name or None if not found
    """
    # Normalize owned_by
    if owned_by:
        owned_by = owned_by.lower()
    
    # Try exact match with owned_by
    if owned_by:
        key = (model, owned_by)
        if key in ROUTING_CONFIG:
            service = ROUTING_CONFIG[key]
            if service:
                logger.info(f"Matched model '{model}' with owned_by '{owned_by}' to service '{service}'")
                return service
    
    # Try match with None (default/any owned_by)
    key = (model, None)
    if key in ROUTING_CONFIG:
        service = ROUTING_CONFIG[key]
        if service:
            logger.info(f"Matched model '{model}' (default owned_by) to service '{service}'")
            return service
    
    # Try default vllm
    key = (model, "vllm")
    if key in ROUTING_CONFIG:
        service = ROUTING_CONFIG[key]
        if service:
            logger.info(f"Matched model '{model}' (default vllm) to service '{service}'")
            return service
    
    # Fuzzy match - try to find similar model names
    model_lower = model.lower()
    for (key_model, key_owned_by), service in ROUTING_CONFIG.items():
        if service is None:
            continue
        
        # Check if model matches (fuzzy)
        key_model_lower = key_model.lower() if key_model else ""
        if key_model_lower in model_lower or model_lower in key_model_lower:
            # If owned_by matches or key_owned_by is None (default)
            if (key_owned_by is None) or (owned_by and key_owned_by == owned_by.lower()):
                logger.info(f"Matched model '{model}' (fuzzy) with owned_by '{owned_by}' to service '{service}'")
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
    target_url = f"http://{service_name}.{NAMESPACE}.svc.cluster.local:{SERVICE_PORT}{path}"
    
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
    return {"status": "healthy", "service": "llmd-api-gateway", "namespace": NAMESPACE}
@app.post("/admin/discover")
async def rediscover_services():
    """Manually trigger service discovery"""
    discovered = await discover_services()
    return {"message": "Service discovery completed", "discovered": len(discovered), "services": discovered}
@app.get("/admin/api/routing", response_model=List[RoutingMappingResponse])
async def get_routing_config():
    """Get all routing mappings"""
    mappings = []
    for (model, owned_by), service_name in ROUTING_CONFIG.items():
        mappings.append(RoutingMappingResponse(
            model=model,
            owned_by=owned_by,
            service_name=service_name
        ))
    return mappings
@app.post("/admin/api/routing", response_model=RoutingMappingResponse)
async def add_routing_mapping(mapping: RoutingMapping):
    """Add a new routing mapping"""
    key = (mapping.model, mapping.owned_by)
    if key in ROUTING_CONFIG:
        raise HTTPException(
            status_code=400, 
            detail=f"Mapping already exists for model '{mapping.model}' with owned_by '{mapping.owned_by}'"
        )
    
    ROUTING_CONFIG[key] = mapping.service_name
    logger.info(f"Added routing mapping: {key} -> {mapping.service_name}")
    
    return RoutingMappingResponse(
        model=mapping.model,
        owned_by=mapping.owned_by,
        service_name=mapping.service_name
    )
@app.delete("/admin/api/routing")
async def delete_routing_mapping(model: str, owned_by: Optional[str] = None):
    """Delete a routing mapping"""
    if owned_by == "":
        owned_by = None
    
    key = (model, owned_by)
    if key not in ROUTING_CONFIG:
        raise HTTPException(
            status_code=404, 
            detail=f"Mapping not found for model '{model}' with owned_by '{owned_by}'"
        )
    
    del ROUTING_CONFIG[key]
    logger.info(f"Deleted routing mapping: {key}")
    
    return {"message": "Mapping deleted successfully"}

@app.get("/v1/models")
async def list_models():
    """List all available models from all ModelServices
    
    Based on reference implementation but adapted for llm-d with pod querying
    to ensure complete model metadata (created, max_model_len, etc.)
    """
    models = []
    seen_keys = set()  # Track (model_id, owned_by) combinations
    pod_model_cache = {}  # Cache models by pod IP for complete metadata
    
    # First, query pods directly to get complete model information
    # This ensures we get all models with full metadata even if services don't return all
    try:
        pods = v1.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector="llm-d.ai/role=decode"
        )
        for pod in pods.items:
            pod_ip = pod.status.pod_ip
            if not pod_ip:
                continue
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"http://{pod_ip}:{SERVICE_PORT}/v1/models")
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and len(data["data"]) > 0:
                            # Store model info by pod IP - each pod serves one model
                            pod_model_cache[pod_ip] = data["data"][0]
                            logger.info(f"Retrieved model from pod {pod.metadata.name} ({pod_ip}): {data['data'][0].get('id')}")
            except Exception as e:
                logger.warning(f"Failed to query pod {pod.metadata.name} ({pod_ip}): {e}")
    except Exception as e:
        logger.warning(f"Failed to query pods directly: {e}")
    
    # Build mapping: service_name -> list of (model_name, owned_by) tuples
    service_to_mappings = {}
    for (model_name, owned_by), service_name in ROUTING_CONFIG.items():
        if service_name and service_name is not None:
            if service_name not in service_to_mappings:
                service_to_mappings[service_name] = []
            service_to_mappings[service_name].append((model_name, owned_by))
    
    # For each service, get models and match with pod cache for complete metadata
    for service_name, mappings in service_to_mappings.items():
        try:
            # Get models from the service
            async with httpx.AsyncClient(timeout=5.0) as client:
                target_url = f"http://{service_name}.{NAMESPACE}.svc.cluster.local:{SERVICE_PORT}/v1/models"
                response = await client.get(target_url)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        # For each model returned by the service
                        for model in data["data"]:
                            model_id = model.get("id", "")
                            if model_id:
                                # Find matching mappings for this model
                                for model_name, owned_by in mappings:
                                    # Check if this model matches the mapping (fuzzy matching)
                                    if model_id == model_name or model_name in model_id or model_id in model_name:
                                        # Create unique key: (model_id, owned_by)
                                        key = (model_id, owned_by)
                                        if key not in seen_keys:
                                            seen_keys.add(key)
                                            engine_name = owned_by or "vllm"
                                            
                                            # Try to find this model in pod cache for complete metadata
                                            model_entry = None
                                            for pod_ip, pod_model in pod_model_cache.items():
                                                pod_model_id = pod_model.get("id", "")
                                                if pod_model_id == model_id:
                                                    # Use pod model for complete metadata
                                                    model_entry = pod_model.copy()
                                                    break
                                            
                                            # If not found in pod cache, use service model
                                            if model_entry is None:
                                                model_entry = model.copy()
                                            
                                            # Clean up and set owned_by
                                            model_entry["id"] = model_id
                                            model_entry.pop("permission", None)
                                            model_entry.pop("engine", None)
                                            model_entry["owned_by"] = engine_name
                                            # Preserve ALL fields: created, max_model_len, root, parent, etc.
                                            models.append(model_entry)
                                            logger.info(f"Added model {model_id} (owned_by: {engine_name}): created={model_entry.get('created')}, max_model_len={model_entry.get('max_model_len')}")
        except Exception as e:
            logger.warning(f"Failed to get models from {service_name}: {e}")
    
    # Also add any models from pod cache that weren't matched in routing config
    # This ensures we capture ALL models with complete metadata
    for pod_ip, pod_model in pod_model_cache.items():
        pod_model_id = pod_model.get("id", "")
        if pod_model_id:
            # Check if this model is already in our list
            already_added = any(m.get("id") == pod_model_id for m in models)
            if not already_added:
                # Find matching owned_by from routing config, or use default
                owned_by = "vllm"
                for (config_model, config_owned_by), config_svc in ROUTING_CONFIG.items():
                    if (config_model == pod_model_id or 
                        pod_model_id.lower() in config_model.lower() or 
                        config_model.lower() in pod_model_id.lower()):
                        owned_by = config_owned_by or "vllm"
                        break
                
                key = (pod_model_id, owned_by)
                if key not in seen_keys:
                    seen_keys.add(key)
                    model_entry = pod_model.copy()
                    model_entry["id"] = pod_model_id
                    model_entry.pop("permission", None)
                    model_entry.pop("engine", None)
                    model_entry["owned_by"] = owned_by
                    models.append(model_entry)
                    logger.info(f"Added model {pod_model_id} from pod cache (not in routing config): created={model_entry.get('created')}, max_model_len={model_entry.get('max_model_len')}")
    
    # If no models found, add fallback from routing config
    if not models:
        for (model_name, owned_by), service_name in ROUTING_CONFIG.items():
            if model_name and service_name is not None:
                key = (model_name, owned_by)
                if key not in seen_keys:
                    seen_keys.add(key)
                    engine_name = owned_by or "vllm"
                    models.append({
                        "id": model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": engine_name
                    })
    
    logger.info(f"Returning {len(models)} models total")
    return {"object": "list", "data": models}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE"])
    async def proxy_v1(request: Request, path: str):
        """Proxy all /v1/* requests (except /v1/models which is handled above)"""
        # Skip /v1/models as it's handled by list_models() above
        if path == "models" and request.method == "GET":
            # This should not happen as list_models() should handle it, but just in case
            return await list_models()
        
        # Read request body
        body = await request.body()
    
        # Parse request body to get model and owned_by fields
        model = None
        owned_by = None
    
        if body:
            try:
                body_json = json.loads(body)
                model = body_json.get("model")
                # Check for owned_by, engine, or inference_server field
                owned_by = (body_json.get("owned_by") or 
                           body_json.get("engine") or 
                           body_json.get("inference_server"))
            except json.JSONDecodeError:
                pass
    
        # If no model from body, try query parameters
        if not model:
            model = request.query_params.get("model")
    
        # If no owned_by from body, try header or query parameters
        if not owned_by:
            owned_by = (request.headers.get("x-owned-by") or
                       request.headers.get("X-Owned-By") or
                       request.query_params.get("owned_by") or
                       request.query_params.get("engine") or
                       request.query_params.get("inference_server"))
    
        # If still no model, return error
        if not model:
            raise HTTPException(
                status_code=400,
                detail="Missing 'model' field in request body, query parameters, or headers"
            )
    
    # Get corresponding Service based on model and owned_by
    service_name = get_service_for_model_and_owned_by(model, owned_by)
    if not service_name:
        # Try to discover services again (async call in sync context - will be handled by admin API)
        # For now, just log and continue
        logger.warning(f"Service not found for model '{model}' with owned_by '{owned_by}'. Try calling POST /admin/discover first.")
        service_name = get_service_for_model_and_owned_by(model, owned_by)
    
    if not service_name:
            available_models = [f"{m} (owned_by: {o})" if o else m 
                               for (m, o) in ROUTING_CONFIG.keys() if m]
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model}' with owned_by '{owned_by}' not found. "
                       f"Available: {available_models}. "
                       f"Try calling POST /admin/discover to refresh service discovery."
            )
    
    logger.info(f"Routing request for model '{model}' with owned_by '{owned_by}' to service '{service_name}'")
    
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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
