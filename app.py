import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from api.main import app

if __name__ == "__main__":
    import uvicorn
    import os
    # Use PORT environment variable if available (Railway), else default to 7860 (HF Spaces)
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
