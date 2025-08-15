import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import threading
import signal
import time
import traceback

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

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more verbose logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Create Flask app for health checks
app = Flask(__name__)

# Global variable to track agent status
agent_status = {
    "initialized": False,
    "running": False,
    "error": None,
    "last_heartbeat": None,
    "start_attempts": 0,
    "last_error_traceback": None
}

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for speech-to-text, LLM, and text-to-speech',
        'LIVEKIT_API_KEY': 'LiveKit API key for room access',
        'LIVEKIT_API_SECRET': 'LiveKit API secret for authentication',
        'LIVEKIT_URL': 'LiveKit server URL'
    }

    missing_vars = []
    env_status = {}
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value == f'your_{var.lower()}_here':
            missing_vars.append(f'{var} ({description})')
            env_status[var] = "MISSING"
        else:
            env_status[var] = "SET" if not value.startswith('your_') else "PLACEHOLDER"
            logger.info(f"{var}: {'✓' if env_status[var] == 'SET' else '✗'}")

    if missing_vars:
        error_msg = (
            "Missing or invalid environment variables:\n" +
            "\n".join(f" - {var}" for var in missing_vars) +
            "\n\nPlease set your environment variables in your deployment dashboard.\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return env_status

class FitnessAssistant:
    """
    AndrofitAI: An energetic, voice-interactive, and supportive AI personal gym coach.
    """

    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.room = ctx.room
        
        try:
            # Initialize components
            logger.info("Initializing STT...")
            self.stt_instance = openai.STT(model="whisper-1")
            
            logger.info("Initializing LLM...")
            self.llm_instance = openai.LLM(model="gpt-4o-mini", temperature=0.7)
            
            logger.info("Initializing TTS...")
            self.tts_instance = openai.TTS(model="tts-1", voice="alloy")
            
            logger.info("Loading VAD model...")
            self.vad_instance = silero.VAD.load()
            
            logger.info("FitnessAssistant initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing FitnessAssistant components: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def start(self):
        """Start the fitness assistant."""
        try:
            logger.info(f"Connecting to room: {self.room.name}")
            # Connect to the room
            await self.ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
            logger.info(f"Successfully connected to room: {self.room.name}")

            # Initial greeting
            logger.info("Sending initial greeting...")
            await self.say_greeting()
            
            # Start listening for audio
            logger.info("Setting up audio pipeline...")
            await self.setup_audio_pipeline()
            
            # Update agent status
            agent_status["running"] = True
            agent_status["last_heartbeat"] = time.time()
            
            logger.info("FitnessAssistant is ready and listening...")
            
        except Exception as e:
            logger.error(f"Error starting FitnessAssistant: {str(e)}")
            logger.error(traceback.format_exc())
            agent_status["error"] = str(e)
            agent_status["last_error_traceback"] = traceback.format_exc()
            raise

    async def say_greeting(self):
        """Send initial greeting."""
        greeting = (
            "Hey there! I'm AndrofitAI, your personal gym coach! "
            "How's your vibe today? Ready to crush it? "
            "Tell me about your fitness goals, experience level, and what equipment you have available!"
        )
        
        await self.say(greeting)

    async def say(self, text: str):
        """Convert text to speech and publish to room."""
        try:
            logger.info(f"AndrofitAI: {text}")
            
            # Generate audio using TTS
            tts_stream = self.tts_instance.synthesize(text)
            audio_frames = []
            
            async for frame in tts_stream:
                audio_frames.append(frame)
            
            # If we have audio frames, publish them
            if audio_frames:
                # Create audio track and publish
                source = rtc.AudioSource(24000, 1)  # 24kHz mono
                track = rtc.LocalAudioTrack.create_audio_track("assistant_voice", source)
                
                options = rtc.TrackPublishOptions()
                options.source = rtc.TrackSource.SOURCE_MICROPHONE
                
                publication = await self.room.local_participant.publish_track(track, options)
                logger.info(f"Published audio track: {publication.sid}")
                
                # Push audio frames
                for frame in audio_frames:
                    await source.capture_frame(frame)
                    
        except Exception as e:
            logger.error(f"Error in TTS: {str(e)}")
            logger.error(traceback.format_exc())

    async def setup_audio_pipeline(self):
        """Set up the audio processing pipeline."""
        try:
            # Subscribe to audio tracks from participants
            for participant in self.room.remote_participants.values():
                await self.subscribe_to_participant(participant)
            
            # Listen for new participants
            @self.room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.info(f"Participant connected: {participant.identity}")
                asyncio.create_task(self.subscribe_to_participant(participant))
            
            # Listen for track subscribed events
            @self.room.on("track_subscribed")  
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant
            ):
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"Subscribed to audio track from {participant.identity}")
                    asyncio.create_task(self.process_audio_track(track))
            
        except Exception as e:
            logger.error(f"Error setting up audio pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def subscribe_to_participant(self, participant: rtc.RemoteParticipant):
        """Subscribe to a participant's tracks."""
        try:
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_AUDIO:
                    await publication.set_subscribed(True)
                    
        except Exception as e:
            logger.error(f"Error subscribing to participant {participant.identity}: {str(e)}")
            logger.error(traceback.format_exc())

    async def process_audio_track(self, track: rtc.AudioTrack):
        """Process incoming audio track for speech recognition."""
        try:
            # Create STT stream
            stt_stream = self.stt_instance.recognize(
                audio_stream=track,
                language="en"
            )
            
            # Process STT events
            async for event in stt_stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    transcript = event.alternatives[0].text
                    logger.info(f"User said: {transcript}")
                    
                    # Generate and speak response
                    await self.generate_response(transcript)
                    
                    # Update heartbeat
                    agent_status["last_heartbeat"] = time.time()
                    
        except Exception as e:
            logger.error(f"Error processing audio track: {str(e)}")
            logger.error(traceback.format_exc())

    async def generate_response(self, user_text: str):
        """Generate and speak response to user input."""
        try:
            # System prompt for fitness coaching
            system_prompt = """You are AndrofitAI, an energetic, voice-interactive, and supportive AI personal gym coach.

Guidelines:
- Be enthusiastic and motivational
- Provide personalized workout plans based on user's goals, experience, and equipment
- Give clear, step-by-step exercise instructions
- Offer form tips and safety advice
- Adapt to user's energy level and feedback
- Keep responses conversational and engaging (60-100 words max)
- Use encouraging language like "You've got this!" and "Let's crush it!"
- For workout plans, be specific about reps, sets, and rest periods
- Always prioritize safety and proper form

Respond to the user's message in a way that's helpful for their fitness journey."""

            # Generate response
            chat_ctx = llm.ChatContext()
            chat_ctx.messages.append(llm.ChatMessage(role="system", content=system_prompt))
            chat_ctx.messages.append(llm.ChatMessage(role="user", content=user_text))
            
            response = await self.llm_instance.achat(chat_ctx=chat_ctx)
            response_text = response.choices[0].message.content
            
            # Speak the response
            await self.say(response_text)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(traceback.format_exc())
            await self.say("Sorry, I had a technical hiccup there. Can you repeat that?")

# Flask routes with more detailed status info
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "AndrofitAI Agent",
        "message": "LiveKit agent service is running",
        "agent_status": agent_status,
        "timestamp": time.time()
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok" if agent_status.get("running", False) else "starting",
        "agent_initialized": agent_status.get("initialized", False),
        "agent_running": agent_status.get("running", False),
        "last_heartbeat": agent_status.get("last_heartbeat"),
        "error": agent_status.get("error"),
        "start_attempts": agent_status.get("start_attempts", 0),
        "timestamp": time.time()
    })

@app.route('/status', methods=['GET'])
def status():
    openai_key = os.getenv('OPENAI_API_KEY', '')
    livekit_url = os.getenv('LIVEKIT_URL', '')
    livekit_key = os.getenv('LIVEKIT_API_KEY', '')
    livekit_secret = os.getenv('LIVEKIT_API_SECRET', '')
    
    return jsonify({
        "agent": "AndrofitAI",
        "environment": {
            "openai_configured": bool(openai_key and not openai_key.startswith('your_')),
            "livekit_configured": bool(livekit_url and not livekit_url.startswith('wss://your-')),
            "livekit_key_configured": bool(livekit_key and not livekit_key.startswith('your_')),
            "livekit_secret_configured": bool(livekit_secret and not livekit_secret.startswith('your_')),
            "python_version": sys.version,
        },
        "agent_status": agent_status,
        "last_error_traceback": agent_status.get("last_error_traceback"),
        "timestamp": time.time()
    })

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint with detailed environment info."""
    try:
        env_status = validate_environment()
        return jsonify({
            "environment_validation": "PASSED",
            "environment_status": env_status,
            "agent_status": agent_status,
            "last_error": agent_status.get("error"),
            "last_traceback": agent_status.get("last_error_traceback"),
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({
            "environment_validation": "FAILED",
            "error": str(e),
            "agent_status": agent_status,
            "timestamp": time.time()
        }), 500

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent."""
    logger.info(f"Agent job started for room: {ctx.room.name}")
    
    try:
        # Update agent status
        agent_status["initialized"] = True
        agent_status["error"] = None
        agent_status["last_error_traceback"] = None
        
        # Create and start the fitness assistant
        logger.info("Creating FitnessAssistant instance...")
        assistant = FitnessAssistant(ctx)
        
        logger.info("Starting FitnessAssistant...")
        await assistant.start()
        
        # Keep the agent running
        logger.info("Agent running, starting heartbeat loop...")
        while True:
            await asyncio.sleep(10)  # Heartbeat every 10 seconds
            agent_status["last_heartbeat"] = time.time()
            logger.debug("Agent heartbeat")
            
    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}")
        logger.error(traceback.format_exc())
        agent_status["error"] = str(e)
        agent_status["last_error_traceback"] = traceback.format_exc()
        agent_status["running"] = False
        raise

def run_agent():
    """Run the LiveKit agent."""
    try:
        logger.info("Starting AndrofitAI LiveKit agent...")
        agent_status["start_attempts"] += 1
        
        # Validate environment
        logger.info("Validating environment variables...")
        env_status = validate_environment()
        logger.info("Environment validation passed!")
        
        # Log configuration status
        openai_configured = env_status.get('OPENAI_API_KEY') == 'SET'
        livekit_configured = env_status.get('LIVEKIT_URL') == 'SET'
        livekit_key_configured = env_status.get('LIVEKIT_API_KEY') == 'SET'
        livekit_secret_configured = env_status.get('LIVEKIT_API_SECRET') == 'SET'
        
        logger.info(f"OpenAI API Key configured: {'✓' if openai_configured else '✗'}")
        logger.info(f"LiveKit URL configured: {'✓' if livekit_configured else '✗'}")
        logger.info(f"LiveKit API Key configured: {'✓' if livekit_key_configured else '✗'}")
        logger.info(f"LiveKit API Secret configured: {'✓' if livekit_secret_configured else '✗'}")
        
        if not all([openai_configured, livekit_configured, livekit_key_configured, livekit_secret_configured]):
            raise ValueError("Required API keys not properly configured")

        # Create worker options
        worker_opts = WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )

        # Run the agent
        logger.info("Starting LiveKit worker...")
        logger.info(f"Connecting to: {os.getenv('LIVEKIT_URL')}")
        cli.run_app(worker_opts)
        
    except Exception as e:
        logger.error(f"Failed to start agent: {str(e)}")
        logger.error(traceback.format_exc())
        agent_status["error"] = str(e)
        agent_status["last_error_traceback"] = traceback.format_exc()
        agent_status["initialized"] = False
        agent_status["running"] = False
        
        # Don't exit - keep Flask running for health checks
        logger.info("Agent failed to start, but keeping Flask server running for debugging")

def start_agent_thread():
    """Start the agent in a separate thread with better error handling."""
    def run_with_error_handling():
        try:
            logger.info("Agent thread starting...")
            run_agent()
        except Exception as e:
            logger.error(f"Agent thread crashed: {str(e)}")
            logger.error(traceback.format_exc())
            agent_status["error"] = f"Agent thread crashed: {str(e)}"
            agent_status["last_error_traceback"] = traceback.format_exc()
    
    agent_thread = threading.Thread(
        target=run_with_error_handling, 
        name="LiveKitAgent",
        daemon=True
    )
    agent_thread.start()
    logger.info("Agent thread started")
    return agent_thread

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler) 
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            # Run only the agent
            run_agent()
        elif sys.argv[1] == "download-files":
            logger.info("Pre-loading model files...")
            try:
                silero.VAD.load()
                logger.info("Model files loaded successfully!")
            except Exception as e:
                logger.warning(f"Could not pre-load models: {e}")
                logger.warning(traceback.format_exc())
        elif sys.argv[1] == "flask-only":
            # Run only Flask (for debugging)
            port = int(os.environ.get("PORT", 10000))
            logger.info(f"Starting Flask-only mode on port {port}")
            app.run(host="0.0.0.0", port=port, debug=False)
        elif sys.argv[1] == "debug":
            # Debug mode - validate environment and exit
            try:
                env_status = validate_environment()
                logger.info("Environment validation successful!")
                logger.info(f"Environment status: {env_status}")
            except Exception as e:
                logger.error(f"Environment validation failed: {str(e)}")
                sys.exit(1)
        else:
            logger.error(f"Unknown command: {sys.argv[1]}")
            sys.exit(1)
    else:
        # Default: Run both Flask and Agent
        logger.info("Starting in hybrid mode (Flask + LiveKit Agent)")
        
        # Start agent in background thread
        start_agent_thread()
        
        # Wait a moment for agent to initialize
        time.sleep(5)
        
        # Start Flask server
        port = int(os.environ.get("PORT", 10000))
        logger.info(f"Starting Flask server on port {port}")
        
        app.run(
            host="0.0.0.0", 
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True
        )
