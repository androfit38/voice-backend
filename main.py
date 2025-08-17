import os
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
)
# Import the correct turn detector
from livekit.plugins.turn_detector import MultilingualTurnDetector

# Load environment variables from .env file
load_dotenv()

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for speech-to-text, LLM, and text-to-speech',
        'LIVEKIT_API_KEY': 'LiveKit API key for room access',
        'LIVEKIT_API_SECRET': 'LiveKit API secret for authentication',
        'LIVEKIT_URL': 'LiveKit server URL'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var) or os.getenv(var) == f'your_{var.lower()}_here':
            missing_vars.append(f'{var} ({description})')
    
    if missing_vars:
        error_msg = (
            "Missing or invalid environment variables:\n" +
            "\n".join(f"  - {var}" for var in missing_vars) +
            "\n\nPlease create a .env file in the backend/ directory with your API keys.\n" +
            "Copy env.example to .env and fill in your actual API keys."
        )
        raise ValueError(error_msg)

# Validate environment variables on import
validate_environment()

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

async def download_models():
    """Download required models for the agent."""
    try:
        print("Downloading required models...")
        # Download turn detector model
        await MultilingualTurnDetector.load()
        print("Models downloaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to download models: {str(e)}")
        print("The agent may still work but with reduced functionality.")

async def entrypoint(ctx: agents.JobContext):
    try:
        # Initialize turn detector with error handling
        try:
            turn_detector = MultilingualTurnDetector()
        except Exception as e:
            print(f"Warning: Turn detector initialization failed: {str(e)}")
            print("Falling back to basic VAD...")
            turn_detector = silero.VAD.load()
        
        # Initialize session with proper error handling
        session = AgentSession(
            stt=openai.STT(
                model="whisper-1",
            ),
            llm=openai.LLM(
                model="gpt-4o-mini"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="alloy",
                instructions="Speak in a friendly and conversational tone."
            ),
            vad=silero.VAD.load(),
            turn_detection=turn_detector,
        )
    except Exception as e:
        print(f"Failed to initialize agent session: {str(e)}")
        raise

    # Start the session with the FitnessAssistant agent
    await session.start(
        room=ctx.room,
        agent=FitnessAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Greet the user and offer assistance
    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

async def main():
    """Main function with model download support."""
    import sys
    
    # Check if we need to download models
    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        await download_models()
        return
    
    try:
        print("Starting AndrofitAI agent...")
        print(f"OpenAI API Key configured: {'✓' if os.getenv('OPENAI_API_KEY') and not os.getenv('OPENAI_API_KEY').startswith('your_') else '✗'}")
        print(f"LiveKit configured: {'✓' if os.getenv('LIVEKIT_URL') and not os.getenv('LIVEKIT_URL').startswith('wss://your-') else '✗'}")
        
        # Try to download models automatically
        await download_models()
        
        # Run the agent app from the command line
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
                # Add timeout and retry configurations
                ws_url=os.getenv("LIVEKIT_URL"),
                api_key=os.getenv("LIVEKIT_API_KEY"),
                api_secret=os.getenv("LIVEKIT_API_SECRET"),
            )
        )
    except ValueError as e:
        # Handle environment variable errors
        print(f"Configuration Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting agent: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your .env file is properly configured")
        print("2. Check your internet connection")
        print("3. Verify your API keys are valid")
        print("4. Make sure LiveKit server is accessible")
        print("5. Run 'python your_agent.py download-files' to download required models")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
