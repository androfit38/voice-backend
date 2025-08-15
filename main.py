import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import threading
import signal
import time

from livekit import agents, rtc
from livekit.agents import (
    AutoSubscribe, 
    JobContext, 
    WorkerOptions, 
    cli,
    llm,
    stt,
    tts,
    vad
)
from livekit.plugins import openai, silero

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Create Flask app for health checks
app = Flask(__name__)

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
        value = os.getenv(var)
        if not value or value == f'your_{var.lower()}_here':
            missing_vars.append(f'{var} ({description})')

    if missing_vars:
        error_msg = (
            "Missing or invalid environment variables:\n" +
            "\n".join(f" - {var}" for var in missing_vars) +
            "\n\nPlease set your environment variables in your deployment dashboard.\n"
        )
        raise ValueError(error_msg)

class FitnessAssistant:
    """
    AndrofitAI: An energetic, voice-interactive, and supportive AI personal gym coach.
    Guides users through personalized workout sessions with motivational feedback and real-time instructions.
    """

    def __init__(self):
        # Initialize components with error handling
        self._initialize_components()

    def _initialize_components(self):
        """Initialize AI components with proper error handling."""
        try:
            # Initialize STT (Speech-to-Text)
            self.stt = openai.STT(
                model="whisper-1",
                language="en"
            )
            logger.info("STT initialized successfully")

            # Initialize LLM (Large Language Model) 
            self.llm = openai.LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            )
            logger.info("LLM initialized successfully")

            # Initialize TTS (Text-to-Speech)
            self.tts = openai.TTS(
                model="tts-1",
                voice="alloy",
            )
            logger.info("TTS initialized successfully")

            # Initialize VAD (Voice Activity Detection)
            self.vad = silero.VAD.load()
            logger.info("VAD initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    async def say(self, text: str, allow_interruptions: bool = True):
        """Convert text to speech and play it."""
        try:
            logger.info(f"TTS: {text}")
            await self.tts.say(text, allow_interruptions=allow_interruptions)
        except Exception as e:
            logger.error(f"Error in TTS: {str(e)}")

# Flask routes for health checks and status
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "AndrofitAI Agent",
        "message": "LiveKit agent is running",
        "timestamp": time.time()
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": time.time()
    })

@app.route('/status', methods=['GET'])
def status():
    openai_key = os.getenv('OPENAI_API_KEY', '')
    livekit_url = os.getenv('LIVEKIT_URL', '')
    
    return jsonify({
        "agent": "AndrofitAI",
        "environment": {
            "openai_configured": bool(openai_key and not openai_key.startswith('your_')),
            "livekit_configured": bool(livekit_url and not livekit_url.startswith('wss://your-')),
            "python_version": sys.version,
            "agents_version": getattr(agents, '__version__', 'unknown')
        },
        "timestamp": time.time()
    })

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent."""
    logger.info(f"Connecting to room: {ctx.room.name}")
    
    try:
        # Create the fitness assistant
        assistant = FitnessAssistant()
        
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info("Connected to room successfully")

        # Greet the user
        greeting = (
            "Hey there! I'm AndrofitAI, your personal gym coach! "
            "How's your vibe today? Ready to crush it? "
            "Tell me about your fitness goals, experience level, and what equipment you have available!"
        )
        
        await assistant.say(greeting, allow_interruptions=True)

        # Set up the main conversation loop
        chat = rtc.ChatManager(ctx.room)

        async def answer_from_text(text: str):
            """Generate and speak a response to user text."""
            logger.info(f"User said: {text}")
            
            try:
                # Create fitness coach prompt
                system_prompt = """You are AndrofitAI, an energetic, voice-interactive, and supportive AI personal gym coach.

Guidelines:
- Be enthusiastic and motivational
- Provide personalized workout plans based on user's goals, experience, and equipment
- Give clear, step-by-step exercise instructions
- Offer form tips and safety advice
- Adapt to user's energy level and feedback
- Keep responses conversational and engaging
- Use encouraging language like "You've got this!" and "Let's crush it!"
- For workout plans, be specific about reps, sets, and rest periods
- Always prioritize safety and proper form

Respond to the user's message in a way that's helpful for their fitness journey."""

                # Generate response using LLM
                response = await assistant.llm.achat(
                    chat_ctx=llm.ChatContext(
                        messages=[
                            llm.ChatMessage(role="system", content=system_prompt),
                            llm.ChatMessage(role="user", content=text)
                        ]
                    )
                )
                
                # Speak the response
                await assistant.say(response.choices[0].message.content)
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                await assistant.say("Sorry, I had a technical hiccup there. Can you repeat that?")

        # Listen for speech
        async def on_human_speech(ev: stt.SpeechEvent):
            """Handle human speech events."""
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                await answer_from_text(ev.alternatives[0].text)

        # Set up speech recognition
        assistant.stt.on("speech_event", on_human_speech)
        
        # Start listening for audio
        assistant.stt.start()
        
        logger.info("Agent is ready and listening...")
        
        # Keep the agent running
        await asyncio.Event().wait()

    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}")
        raise

def run_agent():
    """Run the LiveKit agent."""
    try:
        logger.info("Starting AndrofitAI agent...")
        
        # Check environment configuration
        openai_configured = bool(os.getenv('OPENAI_API_KEY') and not os.getenv('OPENAI_API_KEY').startswith('your_'))
        livekit_configured = bool(os.getenv('LIVEKIT_URL') and not os.getenv('LIVEKIT_URL').startswith('wss://your-'))
        
        logger.info(f"OpenAI API Key configured: {'✓' if openai_configured else '✗'}")
        logger.info(f"LiveKit configured: {'✓' if livekit_configured else '✗'}")
        
        # Validate environment variables
        validate_environment()

        # Create worker options with timeout settings
        worker_opts = WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            # Add timeout configurations
            job_request_timeout=30.0,
            # Set worker permissions
            permissions=rtc.ParticipantPermissions(
                can_subscribe=True,
                can_publish=True,
                can_publish_data=True,
            ),
        )

        # Run the agent
        cli.run_app(worker_opts)
        
    except ValueError as e:
        logger.error(f"Configuration Error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting agent: {str(e)}")
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Ensure your environment variables are properly configured")
        logger.info("2. Check your internet connection")
        logger.info("3. Verify your API keys are valid")
        logger.info("4. Make sure LiveKit server is accessible")
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            # If called with "start" argument, just run the agent
            run_agent()
        elif sys.argv[1] == "download-files":
            # Handle download-files command for Docker build
            logger.info("Downloading model files...")
            try:
                # Pre-load models
                silero.VAD.load()
                logger.info("Model files downloaded successfully!")
            except Exception as e:
                logger.warning(f"Could not download model files: {e}")
                logger.info("This is normal during build - files will be downloaded at runtime.")
        else:
            logger.error(f"Unknown command: {sys.argv[1]}")
            logger.info("Available commands: start, download-files")
            sys.exit(1)
    else:
        # Default behavior: start both Flask and agent
        logger.info("Starting in hybrid mode (Flask + Agent)")
        
        # Start the agent in a background thread
        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()
        
        # Give the agent thread a moment to start
        time.sleep(2)

        # Run Flask app on the port provided by the platform
        port = int(os.environ.get("PORT", 10000))
        logger.info(f"Starting Flask server on port {port}")
        
        try:
            app.run(
                host="0.0.0.0", 
                port=port,
                debug=False,
                use_reloader=False,  # Disable reloader to prevent conflicts
                threaded=True
            )
        except Exception as e:
            logger.error(f"Flask server error: {str(e)}")
            sys.exit(1)
