import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from config.settings import get_settings
from generation.llm_wrapper import LLMWrapper

def verify_openai_switch():
    print("Verifying OpenAI Configuration Switch...")
    settings = get_settings()
    print(f"Provider: {settings.llm.provider}")
    print(f"Model: {settings.llm.model_name}")
    
    if settings.llm.provider != "openai":
        print("FAIL: Provider is not openai")
        return
        
    if settings.llm.model_name != "gpt-4o":
        print("FAIL: Model is not gpt-4o")
        return

    print("\nInitializing LLM Wrapper...")
    try:
        # Check if API key is present
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not found in environment. LLM initialization might fail if not passed explicitly.")
        else:
            print("OPENAI_API_KEY found in environment.")

        llm = LLMWrapper()
        print("LLM Wrapper initialized successfully.")
        
        # Check internal client type
        from langchain_openai import ChatOpenAI
        if isinstance(llm.llm, ChatOpenAI):
             print("SUCCESS: Underlying LLM is ChatOpenAI")
        else:
             print(f"FAIL: Underlying LLM is {type(llm.llm)}")

    except Exception as e:
        print(f"Error initializing LLM: {e}")

if __name__ == "__main__":
    verify_openai_switch()
