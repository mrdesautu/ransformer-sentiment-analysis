#!/usr/bin/env python3
"""
Simple HTTP server to serve the web interface for the Transformer Sentiment Analysis project.
"""

import http.server
import socketserver
import os
import webbrowser
import argparse
from pathlib import Path

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support."""
    
    def end_headers(self):
        """Add CORS headers to allow API requests."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight OPTIONS requests."""
        self.send_response(200)
        self.end_headers()

def serve_web_interface(port=8080, open_browser=True):
    """
    Serve the web interface on the specified port.
    
    Args:
        port (int): Port to serve on
        open_browser (bool): Whether to open browser automatically
    """
    # Change to web directory
    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        print(f"❌ Web directory not found: {web_dir}")
        return
    
    os.chdir(web_dir)
    
    # Create server
    handler = CORSHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    
    print(f"🌐 Serving web interface at: http://localhost:{port}")
    print(f"📁 Serving from: {web_dir}")
    print("📋 Available endpoints:")
    print("   • http://localhost:8080         - Web Interface")
    print("   • http://localhost:8000/health  - API Health Check")
    print("   • http://localhost:8000/docs    - API Documentation")
    print("\n⚡ To test the complete system:")
    print("1. Start API: python -m src.api --host 127.0.0.1 --port 8000")
    print("2. Start Web: python serve_web.py")
    print("3. Open: http://localhost:8080")
    
    if open_browser:
        print(f"\n🚀 Opening browser...")
        webbrowser.open(f"http://localhost:{port}")
    
    print(f"\n🔄 Server running... Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
        httpd.shutdown()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Serve Transformer Sentiment Analysis web interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    serve_web_interface(port=args.port, open_browser=not args.no_browser)

if __name__ == "__main__":
    main()