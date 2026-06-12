import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    print("==================================================================")
    print("             BORUSZYN COAL STOCKPILE DIGITAL TWIN SERVER          ")
    print("==================================================================")
    print(f"  - Project Root: {DIRECTORY}")
    print(f"  - Local Server Link: http://localhost:{PORT}")
    print("  - Serving interactive 3D model: output/mesh_calibrated.ply")
    print("  - Serving calibration database: output/volume_data.json")
    print("------------------------------------------------------------------")
    print("Press Ctrl+C to terminate the dashboard server.")
    print("==================================================================")
    
    # Enable socket reuse to prevent port-in-use errors on restarts
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", PORT), DashboardRequestHandler) as httpd:
            # Open browser window automatically
            webbrowser.open(f"http://localhost:{PORT}")
            # Start loop
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n--> Dashboard server successfully shut down. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n--> Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
