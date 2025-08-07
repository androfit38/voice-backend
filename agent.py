import os
import asyncio
import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    silero,
)

# Load environment variables from .env file first
load_dotenv()

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # More verbose logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Conditional import for noise cancellation
try:
    from livekit.plugins import noise_cancellation
    NOISE_CANCELLATION_AVAILABLE = True
    logger.info("Noise cancellation plugin loaded successfully")
except ImportError as e:
    NOISE_CANCELLATION_AVAILABLE = False
    logger.warning(f"Noise cancellation plugin not available: {e}")

# Conditional import for turn detector
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    TURN_DETECTOR_AVAILABLE = True
    logger.info("Turn detector plugin loaded successfully")
except ImportError as e:
    TURN_DETECTOR_AVAILABLE = False
    logger.warning(f"Turn detector plugin not available: {e}")

class FitnessAssistant(Agent):
    """
    AndrofitAI: An energetic, voice-interactive, and supportive AI personal gym coach.
    Guides users through personalized workout sessions with motivational feedback and real-time instructions.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are AndrofitAI, an energetic, voice-interactive, and supportive AI personal gym coach. "
                "Start every workout session with a warm, personal greeting like 'How's your vibe today? Ready to crush it?' "
                "Prompt users to share their fitness goals, experience level, available equipment, and time, then dynamically generate customized workout plans — "
                "For example, if a user says, 'Beginner, 20 min, no equipment,' offer a suitable plan such as '20-min bodyweight HIIT: 10 squats, 10 push-ups.' "
                "Guide workouts in real time with step-by-step verbal instructions, providing clear cues for each exercise, set, rep, and rest interval — "
                "Support voice commands like 'Pause,' 'Skip,' or 'Make it easier' to ensure users feel in control. "
                "Consistently deliver motivational, context-aware feedback—if a user expresses fatigue, reassure them with, 'You're tough, just two more!' "
                "Share essential form and technique tips by describing correct posture and alignment, and confidently answer questions like 'How's a deadlift done?' "
                "Adopt an authentic personal trainer style: build rapport with empathetic, conversational exchanges and respond to user mood or progress. "
                "During rest intervals, initiate brief, engaging fitness discussions—for example, 'Protein aids recovery; try eggs post-workout.' "
                "Accurately count reps using user grunts, or offer a motivating cadence to keep users on pace, cheering them through every set. "
                "Always focus on making each session positive, safe, goal-oriented, and truly personalized."
            )
        )

async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint for the agent"""
    try:
        logger.info("=== Starting agent session ===")
        logger.info(f"Room ID: {ctx.room.sid if ctx.room else 'None'}")
        
        # Verify required environment variables
        required_vars = ['OPENAI_API_KEY']
        optional_vars = ['LIVEKIT_URL', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET']
        
        # Check required vars
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            raise ValueError(f"Missing environment variables: {missing_vars}")
        
        # Check optional vars (LiveKit might be configured differently in some deployments)
        for var in optional_vars:
            if os.getenv(var):
                logger.info(f"{var} is set")
            else:
                logger.warning(f"{var} is not set")
        
        logger.info("Creating session components...")
        
        # Create session components with simpler configuration
        session_kwargs = {
            'stt': openai.STT(
                model="whisper-1",
            ),
            'llm': openai.LLM(
                model="gpt-4o-mini",  # Fixed model name
                temperature=0.7,
            ),
            'tts': openai.TTS(
                model="tts-1",
                voice="alloy",
            ),
        }
        
        # Add VAD with error handling
        try:
            logger.info("Loading VAD...")
            session_kwargs['vad'] = silero.VAD.load()
            logger.info("VAD loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VAD: {e}")
            # Continue without VAD for now
        
        # Add turn detection if available
        if TURN_DETECTOR_AVAILABLE:
            try:
                logger.info("Loading turn detector...")
                session_kwargs['turn_detection'] = MultilingualModel()
                logger.info("Turn detector loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load turn detector: {e}")
        
        logger.info("Creating AgentSession...")
        session = AgentSession(**session_kwargs)
        
        # Prepare room input options with minimal configuration
        room_input_options = RoomInputOptions()
        
        # Only add noise cancellation if explicitly available
        if NOISE_CANCELLATION_AVAILABLE:
            try:
                logger.info("Adding noise cancellation...")
                room_input_options.noise_cancellation = noise_cancellation.BVC()
                logger.info("Noise cancellation added successfully")
            except Exception as e:
                logger.error(f"Failed to add noise cancellation: {e}")
        
        logger.info("Starting agent session with room...")
        
        # Start the session with the FitnessAssistant agent
        await session.start(
            room=ctx.room,
            agent=FitnessAssistant(),
            room_input_options=room_input_options,
        )
        
        logger.info("=== Agent session started successfully ===")
        
        # Wait a bit before sending initial greeting
        await asyncio.sleep(1)
        
        # Initial greeting
        try:
            await session.generate_reply(
                instructions="Greet the user warmly and ask about their fitness goals for today's session."
            )
            logger.info("Initial greeting sent")
        except Exception as e:
            logger.error(f"Failed to send initial greeting: {e}")
        
    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise

def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for STT, LLM, and TTS',
    }
    
    optional_vars = {
        'LIVEKIT_URL': 'LiveKit server URL',
        'LIVEKIT_API_KEY': 'LiveKit API key', 
        'LIVEKIT_API_SECRET': 'LiveKit API secret'
    }
    
    # Check required variables
    missing_required = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_required.append(f"{var} ({description})")
        else:
            logger.info(f"✓ {var} is set")
    
    if missing_required:
        logger.error("Missing REQUIRED environment variables:")
        for var in missing_required:
            logger.error(f"  - {var}")
        return False
    
    # Check optional variables
    missing_optional = []
    for var, description in optional_vars.items():
        if not os.getenv(var):
            missing_optional.append(f"{var} ({description})")
        else:
            logger.info(f"✓ {var} is set")
    
    if missing_optional:
        logger.warning("Missing OPTIONAL environment variables:")
        for var in missing_optional:
            logger.warning(f"  - {var}")
        logger.info("Agent will still attempt to start...")
    
    return True

if __name__ == "__main__":
    try:
        logger.info("=== Starting LiveKit Agent ===")
        logger.info(f"Python version: {os.sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Print all environment variables for debugging (be careful with secrets)
        logger.info("Environment variables:")
        for key in sorted(os.environ.keys()):
            if 'SECRET' in key or 'KEY' in key or 'TOKEN' in key:
                logger.info(f"  {key}: ***HIDDEN***")
            else:
                logger.info(f"  {key}: {os.environ[key]}")
        
        # Validate environment before starting
        if not validate_environment():
            logger.error("Environment validation failed")
            exit(1)
        
        # Test OpenAI connectivity
        try:
            logger.info("Testing OpenAI connectivity...")
            import openai as openai_client
            # This is just a basic test, adjust based on your OpenAI client version
            logger.info("OpenAI client imported successfully")
        except Exception as e:
            logger.error(f"OpenAI connectivity test failed: {e}")
        
        # Create worker options with minimal configuration
        logger.info("Creating worker options...")
        worker_options = agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
        
        logger.info("Starting agent with CLI...")
        logger.info("If this hangs, the issue is likely with LiveKit server connectivity")
        
        # Run the agent app from the command line
        agents.cli.run_app(worker_options)
        
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except Exception as e:
        logger.error(f"Error starting agent: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        exit(1)
