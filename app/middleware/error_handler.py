# app/middleware/error_handler.py
"""
Error Handler Middleware - Global error handling for FastAPI.
Catch all exceptions and return consistent error responses.
"""

import logging
import traceback
import json
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that safely handles non-serializable objects"""
    def default(self, obj):
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        # Fallback: convert everything else to string
        return str(obj)


class SafeJSONResponse(JSONResponse):
    """JSONResponse that safely handles non-serializable objects"""
    def render(self, content):
        def default_handler(obj):
            """Fallback for non-serializable objects"""
            return str(obj)
        
        try:
            return json.dumps(
                content,
                ensure_ascii=False,
                allow_nan=False,
                indent=None,
                separators=(",", ":"),
                default=default_handler,
            ).encode("utf-8")
        except Exception as e:
            logger.error(f"Failed to render JSON even with default handler: {e}")
            # Last resort - return minimal safe response
            fallback = {
                "error": True,
                "message": "Internal server error"
            }
            return json.dumps(fallback, ensure_ascii=False).encode("utf-8")


class ErrorHandler:
    """Global error handler for the application"""
    
    @staticmethod
    async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle HTTP exceptions"""
        logger.error(f"HTTP Exception: {exc}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": True,
                "message": "Internal server error",
                "detail": str(exc)
            }
        )
    
    @staticmethod
    async def validation_exception_handler(
        request: Request, 
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle validation errors"""
        logger.warning(f"Validation Error: {exc}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": True,
                "message": "Invalid request data",
                "details": exc.errors()
            }
        )
    
    @staticmethod
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all other exceptions"""
        logger.error(f"Unhandled Exception: {exc}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": True,
                "message": "An unexpected error occurred",
                "type": type(exc).__name__
            }
        )


def setup_error_handlers(app):
    """Setup error handlers for FastAPI app"""
    from fastapi.exceptions import HTTPException
    import json as json_module
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response
    
    # ASGI Middleware to catch JSON serialization errors
    class JSONSerializationErrorMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                logger.error(f"Middleware caught exception: {type(e).__name__}: {e}")
                logger.error(traceback.format_exc())
                # Return safe JSON response
                return SafeJSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "error": True,
                        "message": "Internal server error",
                        "type": str(type(e).__name__)
                    }
                )
    
    # Add middleware BEFORE exception handlers
    app.add_middleware(JSONSerializationErrorMiddleware)
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.error(f"HTTP {exc.status_code}: {exc.detail}")
        
        return SafeJSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation Error: {exc}")
        
        # Convert errors to simple, JSON-serializable format
        error_details = []
        try:
            for error in exc.errors():
                try:
                    # Extract only serializable parts
                    error_dict = {
                        "field": str(error.get("loc", ())[-1] if error.get("loc") else "unknown"),
                        "message": str(error.get("msg", "Validation error")),
                        "type": str(error.get("type", "unknown"))
                    }
                    error_details.append(error_dict)
                except Exception as e:
                    logger.warning(f"Failed to serialize error detail: {e}")
                    error_details.append({"message": "Error serialization failed"})
        except Exception as e:
            logger.warning(f"Failed to process validation errors: {e}")
            error_details = [{"message": "Failed to process validation errors"}]
        
        return SafeJSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": True,
                "message": "Invalid request data",
                "details": error_details
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled Exception: {exc}")
        logger.error(f"Exception Type: {type(exc).__name__}")
        logger.error(f"Full Traceback:\n{traceback.format_exc()}")
        
        return SafeJSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": True,
                "message": "An unexpected error occurred",
                "type": str(type(exc).__name__),
                "detail": str(exc)
            }
        )
    
    logger.info(" Error handlers setup complete")
