import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
