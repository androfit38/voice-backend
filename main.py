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

# Global variable to track agent status
agent_status = {
    "initialized": False,
    "running": False,
    "error": None,
    "last_heartbeat": None
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
    """

    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.room = ctx.room
        
        # Initialize components
        self.stt_instance = openai.STT(model="whisper-1")
        self.llm_instance = openai.LLM(model="gpt-4o-mini", temperature=0.7)
        self.tts_instance = openai.TTS(model="tts-1", voice="alloy")
        self.vad_instance = silero.VAD.load()
        
        logger.info("FitnessAssistant initialized successfully")

    async def start(self):
        """Start the fitness assistant."""
        try:
            # Connect to the room
            await self.ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
            logger.info(f"Connected to room: {self.room.name}")

            # Initial greeting
            await self.say_greeting()
            
            # Start listening for audio
            await self.setup_audio_pipeline()
            
            # Update agent status
            agent_status["running"] = True
            agent_status["last_heartbeat"] = time.time()
            
            logger.info("FitnessAssistant is ready and listening...")
            
        except Exception as e:
            logger.error(f"Error starting FitnessAssistant: {str(e)}")
            agent_status["error"] = str(e)
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
            raise

    async def subscribe_to_participant(self, participant: rtc.RemoteParticipant):
        """Subscribe to a participant's tracks."""
        try:
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_AUDIO:
                    await publication.set_subscribed(True)
                    
        except Exception as e:
            logger.error(f"Error subscribing to participant {participant.identity}: {str(e)}")

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
            await self.say("Sorry, I had a technical hiccup there. Can you repeat that?")

# Flask routes
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
        },
        "agent_status": agent_status,
        "timestamp": time.time()
    })

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent."""
    logger.info(f"Agent job started for room: {ctx.room.name}")
    
    try:
        # Update agent status
        agent_status["initialized"] = True
        agent_status["error"] = None
        
        # Create and start the fitness assistant
        assistant = FitnessAssistant(ctx)
        await assistant.start()
        
        # Keep the agent running
        while True:
            await asyncio.sleep(10)  # Heartbeat every 10 seconds
            agent_status["last_heartbeat"] = time.time()
            
    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}")
        agent_status["error"] = str(e)
        agent_status["running"] = False
        raise

def run_agent():
    """Run the LiveKit agent."""
    try:
        logger.info("Starting AndrofitAI LiveKit agent...")
        
        # Validate environment
        validate_environment()
        
        # Log configuration status
        openai_configured = bool(os.getenv('OPENAI_API_KEY') and not os.getenv('OPENAI_API_KEY').startswith('your_'))
        livekit_configured = bool(os.getenv('LIVEKIT_URL') and not os.getenv('LIVEKIT_URL').startswith('wss://your-'))
        
        logger.info(f"OpenAI API Key configured: {'✓' if openai_configured else '✗'}")
        logger.info(f"LiveKit configured: {'✓' if livekit_configured else '✗'}")
        
        if not openai_configured or not livekit_configured:
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
        cli.run_app(worker_opts)
        
    except Exception as e:
        logger.error(f"Failed to start agent: {str(e)}")
        agent_status["error"] = str(e)
        agent_status["initialized"] = False
        agent_status["running"] = False
        
        # Don't exit - keep Flask running for health checks
        logger.info("Agent failed to start, but keeping Flask server running for debugging")

def start_agent_thread():
    """Start the agent in a separate thread."""
    agent_thread = threading.Thread(
        target=run_agent, 
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
        elif sys.argv[1] == "flask-only":
            # Run only Flask (for debugging)
            port = int(os.environ.get("PORT", 10000))
            logger.info(f"Starting Flask-only mode on port {port}")
            app.run(host="0.0.0.0", port=port, debug=False)
        else:
            logger.error(f"Unknown command: {sys.argv[1]}")
            sys.exit(1)
    else:
        # Default: Run both Flask and Agent
        logger.info("Starting in hybrid mode (Flask + LiveKit Agent)")
        
        # Start agent in background thread
        start_agent_thread()
        
        # Wait a moment for agent to initialize
        time.sleep(3)
        
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
