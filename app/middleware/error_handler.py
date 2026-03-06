# app/middleware/error_handler.py
"""
Error Handler Middleware - Global error handling for FastAPI.
Catch all exceptions and return consistent error responses.
"""

import logging
import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger(__name__)


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
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.error(f"HTTP {exc.status_code}: {exc.detail}")
        
        return JSONResponse(
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
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": True,
                "message": "Invalid request data",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
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
    
    logger.info(" Error handlers setup complete")
