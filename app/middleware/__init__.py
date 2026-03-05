# app/middleware/__init__.py
"""Middleware package"""

from .error_handler import setup_error_handlers, ErrorHandler

__all__ = ["setup_error_handlers", "ErrorHandler"]
