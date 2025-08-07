import os
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import LiveKit components
try:
    from livekit import agents
    from livekit.agents import JobContext, WorkerOptions
    from livekit.plugins import openai
    logger.info("LiveKit imports successful")
except ImportError as e:
    logger.error(f"Failed to import LiveKit: {e}")
    exit(1)

async def entrypoint(ctx: JobContext):
    """Simple entrypoint that just connects and stays alive"""
    logger.info("=== Agent Starting ===")
    logger.info(f"Room SID: {ctx.room.sid if ctx.room else 'No room'}")
    
    try:
        # Just connect to the room
        await ctx.connect()
        logger.info("Connected to room successfully!")
        
        # Create a simple LLM for testing
        llm = openai.LLM(model="gpt-4o-mini")
        logger.info("LLM created successfully")
        
        # Keep the connection alive
        logger.info("Agent is running... Press Ctrl+C to stop")
        while True:
            await asyncio.sleep(10)
            logger.info("Agent is still alive...")
            
    except Exception as e:
        logger.error(f"Error in entrypoint: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Main entry point"""
    logger.info("=== Starting Simple Fitness Agent ===")
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY is required!")
        exit(1)
    else:
        logger.info("OpenAI API key found ✓")
    
    # Check LiveKit variables (optional)
    livekit_url = os.getenv("LIVEKIT_URL")
    livekit_key = os.getenv("LIVEKIT_API_KEY") 
    livekit_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if livekit_url and livekit_key and livekit_secret:
        logger.info("LiveKit credentials found ✓")
    else:
        logger.warning("LiveKit credentials not found - using defaults")
    
    try:
        # Create worker options
        options = WorkerOptions(entrypoint_fnc=entrypoint)
        
        # Start the agent
        logger.info("Starting agent with CLI...")
        agents.cli.run_app(options)
        
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
