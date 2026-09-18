"""FastAPI web application for InferForge."""

import os

from fastapi import FastAPI, HTTPException


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="InferForge Web API",
        description="Local AI toolkit web interface",
        version="0.2.0",
    )
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "InferForge Web API",
            "version": "0.2.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint for Render."""
        return {"status": "healthy"}
    
    @app.get("/api/models")
    async def list_models():
        """List all registered models."""
        try:
            from inferforge.core.registry import Registry
            reg = Registry()
            models = reg.list()
            return {
                "models": [
                    {
                        "name": m.name,
                        "family": m.family,
                        "size": m.display_size(),
                        "quantization": m.quantization,
                        "source": m.source
                    }
                    for m in models
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/info")
    async def info():
        """Get system information."""
        return {
            "version": "0.2.0",
            "platform": os.name,
            "features": [
                "model_management",
                "model_merging",
                "training",
                "optimization",
                "nexara_language"
            ]
        }
    
    return app

                     
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)