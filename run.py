#!/usr/bin/env python3
"""
Legal AI System Startup Script

This script provides an easy way to start the Legal AI application with proper configuration.
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

def check_requirements():
    """Check if all required dependencies are installed."""
    try:
        import fastapi
        import anthropic
        import chromadb
        import sqlalchemy
        import redis
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_environment():
    """Check if environment variables are set."""
    required_vars = [
        "ANTHROPIC_API_KEY",
        "DATABASE_URL", 
        "SECRET_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file or set these environment variables")
        return False
    
    print("✅ Environment variables are configured")
    return True

def setup_directories():
    """Create required directories if they don't exist."""
    from app.config import settings
    
    directories = [
        settings.document_storage_path,
        settings.template_storage_path,
        settings.chroma_persist_directory
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Required directories created")

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="Legal AI System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--check-only", action="store_true", help="Only check requirements and exit")
    
    args = parser.parse_args()
    
    print("🏛️  Legal AI System Startup")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    if args.check_only:
        print("✅ All checks passed! System is ready to start.")
        sys.exit(0)
    
    print("\n🚀 Starting Legal AI System...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"❤️  Health Check: http://{args.host}:{args.port}/health")
    print("\n" + "=" * 40)
    
    # Start the server
    try:
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Legal AI System...")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 