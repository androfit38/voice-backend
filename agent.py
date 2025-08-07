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

# Conditional import for noise cancellation
try:
    from livekit.plugins import noise_cancellation
    NOISE_CANCELLATION_AVAILABLE = True
except ImportError:
    NOISE_CANCELLATION_AVAILABLE = False
    logging.warning("Noise cancellation plugin not available")

# Conditional import for turn detector
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    TURN_DETECTOR_AVAILABLE = True
except ImportError:
    TURN_DETECTOR_AVAILABLE = False
    logging.warning("Turn detector plugin not available")

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info("Starting agent session...")
        
        # Verify required environment variables
        required_vars = ['OPENAI_API_KEY', 'LIVEKIT_URL', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            raise ValueError(f"Missing environment variables: {missing_vars}")
        
        # Create session components with error handling
        session_kwargs = {
            'stt': openai.STT(
                model="whisper-1",
            ),
            'llm': openai.LLM(
                model="gpt-4o-mini"  # Fixed model name
            ),
            'tts': openai.TTS(
                model="tts-1",
                voice="alloy",
            ),
            'vad': silero.VAD.load(),
        }
        
        # Add turn detection if available
        if TURN_DETECTOR_AVAILABLE:
            session_kwargs['turn_detection'] = MultilingualModel()
        
        session = AgentSession(**session_kwargs)
        
        # Prepare room input options
        room_input_options = RoomInputOptions()
        if NOISE_CANCELLATION_AVAILABLE:
            room_input_options.noise_cancellation = noise_cancellation.BVC()
        
        logger.info("Starting agent session with room...")
        
        # Start the session with the FitnessAssistant agent
        await session.start(
            room=ctx.room,
            agent=FitnessAssistant(),
            room_input_options=room_input_options,
        )
        
        logger.info("Agent session started successfully")
        
        # Initial greeting
        await session.generate_reply(
            instructions="Greet the user warmly and ask about their fitness goals for today's session."
        )
        
    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}")
        raise

def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for STT, LLM, and TTS',
        'LIVEKIT_URL': 'LiveKit server URL',
        'LIVEKIT_API_KEY': 'LiveKit API key',
        'LIVEKIT_API_SECRET': 'LiveKit API secret'
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    if missing:
        logger.error("Missing required environment variables:")
        for var in missing:
            logger.error(f"  - {var}")
        return False
    
    logger.info("All required environment variables are set")
    return True

if __name__ == "__main__":
    try:
        logger.info("Starting LiveKit agent...")
        
        # Validate environment before starting
        if not validate_environment():
            logger.error("Environment validation failed")
            exit(1)
        
        # Create worker options with proper configuration
        worker_options = agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Add worker-specific options
            prewarm_fnc=None,  # Optional: function to run before worker starts
        )
        
        logger.info("Starting agent with CLI...")
        
        # Run the agent app from the command line
        agents.cli.run_app(worker_options)
        
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except Exception as e:
        logger.error(f"Error starting agent: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
