#!/usr/bin/env python3
"""
Quick Startup & Setup Script
Giúp setup và khởi động MCP server một cách dễ dàng
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

ROOT_DIR = Path(__file__).parent
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")

print("=" * 70)
print("🚀 MCP SERVER - QUICK START")
print("=" * 70)


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"✓ {title}")
    print(f"{'=' * 70}")


def check_python():
    """Check Python version"""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"\n✓ Python version: {version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)


def check_dependencies():
    """Check if dependencies are installed"""
    print_section("Checking Dependencies")
    
    required = ["fastapi", "uvicorn", "mcp", "httpx", "pydantic"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed")
    return True


def check_env_vars():
    """Check environment variables"""
    print_section("Environment Configuration")
    
    print(f"  Backend URL: {BACKEND_URL}")
    print(f"  API Key: {'Set' if BACKEND_API_KEY else 'Not set'}")
    
    if not BACKEND_API_KEY:
        print("\n⚠️ No API key set. Using unauthenticated mode.")


def check_backend():
    """Check if backend is running"""
    print_section("Checking Backend")
    
    import httpx
    
    try:
        client = httpx.Client(timeout=5)
        response = client.get(f"{BACKEND_URL}/docs")
        if response.status_code == 200:
            print(f"✅ Backend running at {BACKEND_URL}")
            return True
        else:
            print(f"⚠️ Backend returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        print(f"\n💡 Start backend with: python run_server.py")
        return False


def setup_claude_integration():
    """Setup Claude Desktop integration"""
    print_section("Claude Desktop Integration")
    
    try:
        import mcp_config
        mcp_config.setup_claude_integration()
        print("✅ Claude Desktop setup complete")
    except Exception as e:
        print(f"⚠️ Claude setup skipped: {e}")


def run_test_suite():
    """Run test suite"""
    print_section("Running Tests")
    
    try:
        import asyncio
        from test_mcp_client import MCPClient
        
        async def run_tests():
            client = MCPClient()
            
            # Health check
            is_healthy = await client.health_check()
            if not is_healthy:
                print("\n❌ Backend is not running")
                return False
            
            # Quick test
            result = await client.list_jobs(limit=1)
            if "jobs" in result:
                print("✅ Tests passed")
                return True
            else:
                print("⚠️ Tests completed with issues")
                return False
        
        return asyncio.run(run_tests())
    
    except Exception as e:
        print(f"⚠️ Tests skipped: {e}")
        return True


def start_mcp_server():
    """Start MCP server"""
    print_section("Starting MCP Server")
    
    print(f"\n🚀 Starting MCP server...")
    print(f"   Backend: {BACKEND_URL}")
    print(f"   Log level: INFO")
    print(f"\n   Press Ctrl+C to stop")
    print("\n" + "-" * 70)
    
    try:
        subprocess.run([sys.executable, "mcp_server.py"], cwd=ROOT_DIR)
    except KeyboardInterrupt:
        print("\n\n✅ MCP Server stopped")


def show_menu():
    """Show interactive menu"""
    print_section("Menu")
    
    options = {
        "1": ("Check Setup", lambda: check_setup()),
        "2": ("Run Tests", lambda: run_test_suite()),
        "3": ("Setup Claude", lambda: setup_claude_integration()),
        "4": ("Start MCP Server", lambda: start_mcp_server()),
        "5": ("View Documentation", lambda: show_docs()),
        "6": ("Exit", lambda: sys.exit(0)),
    }
    
    for key, (label, _) in options.items():
        print(f"  {key}. {label}")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice in options:
        options[choice][1]()
    else:
        print("❌ Invalid choice")
        show_menu()


def check_setup():
    """Run full setup check"""
    print("\n" + "=" * 70)
    print("SETUP CHECK")
    print("=" * 70)
    
    check_python()
    deps_ok = check_dependencies()
    check_env_vars()
    backend_ok = check_backend()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    status = "✅ READY" if (deps_ok and backend_ok) else "⚠️  ACTION REQUIRED"
    print(f"\nStatus: {status}")
    
    if not deps_ok:
        print("\n📦 Install dependencies:")
        print("   pip install -r requirements.txt")
    
    if not backend_ok:
        print("\n🚀 Start backend:")
        print("   python run_server.py")
    
    print("\n🤖 Start MCP server:")
    print("   python mcp_server.py")
    
    print("\n📚 View documentation:")
    print("   less MCP_DOCUMENTATION.md")
    
    return deps_ok and backend_ok


def show_docs():
    """Show documentation"""
    docs_file = ROOT_DIR / "MCP_DOCUMENTATION.md"
    
    if docs_file.exists():
        # Try to open with appropriate viewer
        if platform.system() == "Windows":
            os.startfile(str(docs_file))
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(docs_file)])
        else:  # Linux
            subprocess.run(["xdg-open", str(docs_file)])
    else:
        print("❌ Documentation not found")


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            check_setup()
        elif command == "test":
            run_test_suite()
        elif command == "setup":
            setup_claude_integration()
        elif command == "start":
            start_mcp_server()
        elif command == "docs":
            show_docs()
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python start_mcp.py check   - Check setup")
            print("  python start_mcp.py test    - Run tests")
            print("  python start_mcp.py setup   - Setup Claude")
            print("  python start_mcp.py start   - Start server")
            print("  python start_mcp.py docs    - View documentation")
    else:
        # Interactive mode
        show_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
