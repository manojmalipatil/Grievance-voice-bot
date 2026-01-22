import asyncio
import os
import sqlite3
import uuid
import json
import logging
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# LiveKit Imports
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
# Added 'openai' to imports for Ollama support
from livekit.plugins import sarvam, groq, silero, openai 

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GrievanceBot")

# --- Load Language Configuration ---
def load_language_config(config_path="language_config.json"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"[CONFIG] Successfully loaded language config from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"[ERROR] Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"[ERROR] Invalid JSON in config file: {e}")
        raise

LANGUAGE_CONFIG = load_language_config()
SUPPORTED_LANGUAGES = LANGUAGE_CONFIG["supported_languages"]
LANGUAGE_SELECTION_PROMPT = LANGUAGE_CONFIG["language_selection_prompt"]
GRIEVANCE_PROMPTS = LANGUAGE_CONFIG["grievance_prompts"]

# --- Database Manager ---
class DatabaseManager:
    def __init__(self, db_path="grievance.db"):
        self.db_path = db_path
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grievances (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                transcript TEXT NOT NULL,
                language TEXT,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()

    async def save_grievance(self, transcript: str, language: str = "en"):
        if not transcript.strip():
            logger.warning("[DB] Attempted to save empty grievance, skipping")
            return

        def _save():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            record_id = str(uuid.uuid4())
            current_time = datetime.now().isoformat()
            try:
                cursor.execute("""
                    INSERT INTO grievances (id, timestamp, transcript, language, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (record_id, current_time, transcript, language, current_time))
                conn.commit()
                logger.info(f"[DB] Saved grievance ID: {record_id} (Lang: {language})")
            except Exception as e:
                logger.error(f"[DB] Error saving grievance: {e}")
            finally:
                conn.close()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _save)


class GrievanceTracker:
    def __init__(self):
        self.grievance_text = []
        self.selected_language = None
    
    def set_language(self, language_code: str):
        self.selected_language = language_code
        logger.info(f"[TRACKER] Language set to: {language_code}")
    
    def add_user_message(self, text: str):
        self.grievance_text.append(f"Employee: {text}")
    
    def add_agent_message(self, text: str):
        self.grievance_text.append(f"Agent: {text}")
    
    def get_full_grievance(self) -> str:
        return "\n".join(self.grievance_text)
    
    def reset(self):
        """Reset tracker for new conversation"""
        self.grievance_text = []
        logger.info("[TRACKER] Reset grievance text")


async def entrypoint(ctx: JobContext):
    logger.info(f"[ROOM] Connecting to room: {ctx.room.name}")
    
    # Verify Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("[ERROR] GROQ_API_KEY not found in environment variables!")
        raise ValueError("GROQ_API_KEY is required. Add it to your .env file")
    
    db_manager = DatabaseManager()
    await ctx.connect()
    
    grievance_tracker = GrievanceTracker()
    
    # State management
    should_end_call = asyncio.Event()
    language_selected = asyncio.Event()
    selected_lang_code = "en"
    tool_called = {"called": False}
    
    # --- LANGUAGE SELECTION TOOL ---
    @function_tool
    async def select_language(language: Optional[str] = None):
        """
        Select the language based on user preference. 
        Call this immediately after user states their preference.
        
        Args:
            language: The language name the user mentioned (e.g., "english", "hindi", "tamil", "kannada", etc.).
                     IMPORTANT: Use the EXACT language the user mentioned, not a default!
        """
        nonlocal selected_lang_code
        
        if language is None or not language.strip():
            logger.warning("[TOOL] LLM passed None/empty for language. Defaulting to English.")
            language = "english"

        lang_lower = language.lower().strip()
        logger.info(f"[TOOL] Received language selection: '{lang_lower}'")
        
        # Find matching language code with improved matching
        found_code = None
        found_name = "English"
        
        # First: Try exact match on language name (english, hindi, tamil, kannada, etc.)
        for name, data in SUPPORTED_LANGUAGES.items():
            if name.lower() == lang_lower:
                found_code = data["code"]
                found_name = data["name"]
                logger.info(f"[TOOL] Exact match: '{lang_lower}' → {found_name} ({found_code})")
                break
        
        # Second: Try partial match (if user said "I want kannada")
        if found_code is None:
            for name, data in SUPPORTED_LANGUAGES.items():
                if name.lower() in lang_lower or lang_lower in name.lower():
                    found_code = data["code"]
                    found_name = data["name"]
                    logger.info(f"[TOOL] Partial match: '{lang_lower}' contains {name} → {found_name} ({found_code})")
                    break
        
        # Third: Try matching native script
        if found_code is None:
            for name, data in SUPPORTED_LANGUAGES.items():
                native_name = data["name"].lower()
                if native_name in lang_lower or lang_lower in native_name:
                    found_code = data["code"]
                    found_name = data["name"]
                    logger.info(f"[TOOL] Native script match: '{lang_lower}' → {found_name} ({found_code})")
                    break
        
        # Fourth: Try by language code (en, hi, kn, etc.)
        if found_code is None:
            for name, data in SUPPORTED_LANGUAGES.items():
                if data["code"] == lang_lower:
                    found_code = data["code"]
                    found_name = data["name"]
                    logger.info(f"[TOOL] Code match: '{lang_lower}' → {found_name} ({found_code})")
                    break
        
        # Last resort: Default to English
        if found_code is None:
            logger.warning(f"[TOOL] No match found for '{lang_lower}', defaulting to English")
            found_code = "en"
            found_name = "English"
        
        selected_lang_code = found_code
        grievance_tracker.set_language(selected_lang_code)
        tool_called["called"] = True
        
        logger.info(f"[TOOL] Language finalized: {found_name} ({selected_lang_code})")
        language_selected.set()
        
        return ""  # Empty to prevent speaking

    # --- END CALL TOOL ---
    @function_tool
    async def end_call(confirmation: str = "yes"):
        """
        End the grievance collection call.
        Call this only when the user explicitly indicates they are done.
        """
        logger.info(f"[TOOL] end_call triggered")
        should_end_call.set()
        return ""

    # ============================================================================
    # STAGE 1: LANGUAGE SELECTION
    # ============================================================================
    logger.info("[STAGE 1] Starting Language Selection Phase")
    
    # OPTIMIZED VAD SETTINGS
    vad_config_stage1 = silero.VAD.load(
        min_speech_duration=0.1,       # Reduced to catch speech faster
        min_silence_duration=0.5,      # Reasonable pause detection
        activation_threshold=0.5,      # Moderate threshold
        sample_rate=16000,
    )
    
    # GROQ LLM - LLAMA 3.1 8B INSTANT (Fast & Efficient)
    # This model is extremely fast and great for simple tasks like language selection
    llm_config_stage1 = groq.LLM(
        model="llama-3.1-8b-instant",  # Production model, super fast
        temperature=0.3,  # Low temperature for deterministic responses
    )
    
    session_1 = AgentSession(
        vad=vad_config_stage1,
        stt=groq.STT(
            model="whisper-large-v3-turbo"
            #language="en"  # Always listen in English for language selection
        ),
        llm=llm_config_stage1,
        tts=sarvam.TTS(
            target_language_code="en-IN", 
            speaker="anushka"
        ),
    )
    
    language_agent = Agent(
        instructions=LANGUAGE_SELECTION_PROMPT,
        tools=[select_language],
    )

    # Event listener for debugging
    @session_1.on("conversation_item_added")
    def on_item_stage1(event):
        if event.item.text_content:
            logger.info(f"[STAGE 1] {event.item.role}: {event.item.text_content}")

    # Start session - this begins listening immediately
    session_1_task = asyncio.create_task(
        session_1.start(agent=language_agent, room=ctx.room)
    )
    
    # Wait for session to fully initialize
    logger.info("[STAGE 1] Waiting for session to initialize...")
    await asyncio.sleep(1)  # Increased wait time for full initialization
    
    # Send initial greeting without blocking listening
    logger.info("[STAGE 1] Sending initial greeting...")
    try:
        # Use say() instead of generate_reply() to avoid blocking the conversation flow
        await session_1.say(
            "Hello! Which language would you like to speak? We support English, Hindi, Tamil, Kannada, Telugu, Bengali, Marathi, Gujarati, Malayalam, Punjabi, and Odia.",
            allow_interruptions=True,  # Allow user to interrupt
        )
        logger.info("[STAGE 1] ✓ Greeting sent, now listening...")
    except Exception as e:
        logger.error(f"[STAGE 1] Error sending greeting: {e}")
        # Continue anyway, the agent should still work
    
    # Wait for language selection
    logger.info("[STAGE 1] Waiting for language selection...")
    try:
        await asyncio.wait_for(language_selected.wait(), timeout=60.0)
        logger.info(f"[STAGE 1] ✓ Language selected: {selected_lang_code}")
    except asyncio.TimeoutError:
        logger.warning("[STAGE 1] ⚠ Timeout - defaulting to English")
        selected_lang_code = "en"
        grievance_tracker.set_language(selected_lang_code)
    
    if not tool_called["called"]:
        logger.warning("[STAGE 1] ⚠ Tool was never called")
    
    # ============================================================================
    # GRACEFUL SESSION 1 SHUTDOWN
    # ============================================================================
    logger.info("[TRANSITION] Shutting down language selection...")
    
    try:
        await session_1.aclose()
        logger.info("[TRANSITION] ✓ Session 1 closed")
    except Exception as e:
        logger.error(f"[TRANSITION] Error closing session 1: {e}")
    
    session_1_task.cancel()
    try:
        await session_1_task
    except asyncio.CancelledError:
        logger.info("[TRANSITION] ✓ Session 1 task cancelled")
    
    # CRITICAL BUFFER - Allow audio to drain
    logger.info(f"[TRANSITION] Audio buffer flush ({selected_lang_code})...")
    await asyncio.sleep(0.5)
    
    # ============================================================================
    # STAGE 2: GRIEVANCE COLLECTION
    # ============================================================================
    logger.info("[STAGE 2] Starting Grievance Collection Phase")

    prompt = GRIEVANCE_PROMPTS.get(selected_lang_code, GRIEVANCE_PROMPTS["en"])
    
    # Get TTS code
    tts_code = "en-IN"
    for lang_data in SUPPORTED_LANGUAGES.values():
        if lang_data["code"] == selected_lang_code:
            tts_code = lang_data["sarvam_tts_code"]
            logger.info(f"[STAGE 2] Using TTS code: {tts_code}")
            break

    # OPTIMIZED VAD for Stage 2
    vad_config_stage2 = silero.VAD.load(
        min_speech_duration=0.5,
        min_silence_duration=0.8,
        activation_threshold=0.5,
        sample_rate=16000,
    )
    
    # --- CHANGED: OLLAMA LLM (via OpenAI Plugin) ---
    # Replaced Groq with OpenAI plugin pointing to local Ollama
    llm_config_stage2 = groq.LLM(
        model="openai/gpt-oss-120b",  # Production model, super fast
        temperature=0.3,  # Low temperature for deterministic responses
    )

    session_2 = AgentSession(
        vad=vad_config_stage2,
        stt=groq.STT(
            model="whisper-large-v3-turbo", 
            language=selected_lang_code
        ),
        llm=llm_config_stage2,
        tts=sarvam.TTS(
            target_language_code=tts_code, 
            speaker="anushka"
        ),
    )

    grievance_agent = Agent(
        instructions=prompt,
        tools=[end_call],
    )
    
    # Track conversation
    @session_2.on("conversation_item_added")
    def on_item_stage2(event):
        if event.item.text_content:
            logger.info(f"[STAGE 2] {event.item.role}: {event.item.text_content}")
            if event.item.role == "user":
                grievance_tracker.add_user_message(event.item.text_content)
            elif event.item.role == "assistant":
                grievance_tracker.add_agent_message(event.item.text_content)

    session_2_task = asyncio.create_task(
        session_2.start(agent=grievance_agent, room=ctx.room)
    )
    
    # Warmup period
    logger.info("[STAGE 2] Session warmup...")
    await asyncio.sleep(1)
    
    # Generate greeting
    try:
        await session_2.generate_reply()
        logger.info("[STAGE 2] ✓ Initial greeting sent")
    except asyncio.TimeoutError:
        logger.error("[STAGE 2] ⚠ Timeout generating greeting")
    except Exception as e:
        logger.error(f"[STAGE 2] Error: {e}")

    # Wait for completion
    logger.info("[STAGE 2] Collecting grievance...")
    await should_end_call.wait()
    
    # ============================================================================
    # CLEANUP & SAVE
    # ============================================================================
    logger.info("[CLEANUP] Ending session...")
    await asyncio.sleep(3.0)
    
    try:
        await session_2.aclose()
        logger.info("[CLEANUP] ✓ Session 2 closed")
    except Exception as e:
        logger.error(f"[CLEANUP] Error: {e}")
    
    session_2_task.cancel()
    try:
        await session_2_task
    except asyncio.CancelledError:
        logger.info("[CLEANUP] ✓ Session 2 task cancelled")
    
    # Save to database
    full_log = grievance_tracker.get_full_grievance()
    if full_log.strip():
        logger.info(f"[CLEANUP] Saving {len(full_log)} chars to DB")
        await db_manager.save_grievance(full_log, selected_lang_code)
    else:
        logger.warning("[CLEANUP] ⚠ No content to save")
    
    await ctx.disconnect()
    logger.info("[END] ✓ Session complete")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))