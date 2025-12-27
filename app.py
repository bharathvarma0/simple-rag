import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from api.main import app

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Robust port handling
    port_str = os.environ.get("PORT", "7860")
    print(f"Starting application on 0.0.0.0:{port_str}")
    
    # Check for critical env vars
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set!")
        
    try:
        port = int(port_str)
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)
