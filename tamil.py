import sqlite3
import json
import requests
from datetime import datetime
from deep_translator import GoogleTranslator

# =========================
# Configuration
# =========================
DB_PATH = "grievance.db"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:14b"

# =========================
# Translation Function
# =========================
def translate_to_english(text, language):
    """Translate text to English using Deep Translator (Google)."""
    if not text:
        return text

    if language and language.lower() in ["en", "english"]:
        return text

    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        print(f"Translation error (Deep Translator): {e}")
        return text


# =========================
# Database Schema Update
# =========================
def ensure_columns_exist(conn):
    """Ensure all required columns exist in the grievances table."""
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(grievances)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    new_columns = {
        "translated_transcript": "TEXT",
        "category": "TEXT",
        "priority": "TEXT",
        "sentiment": "TEXT",
        "summary": "TEXT",
        "tags": "TEXT",
        "department": "TEXT",
        "processed_at": "TEXT",
        "analysis_json": "TEXT"
    }

    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE grievances ADD COLUMN {col_name} {col_type}"
            )
            print(f"Added column: {col_name}")

    conn.commit()


# =========================
# Ollama Interaction
# =========================
def send_to_ollama(transcript):
    """Send transcript to Ollama for structured grievance analysis."""
    prompt = f"""
Analyze this customer grievance and provide a structured categorization.

Grievance transcript:
"{transcript}"

Please provide:
1. Category (e.g., POSH, Managerial, Data, Hygiene, Compensation, Workplace Environment, Conflict, Career, Attendance)
2. Priority (High, Medium, Low)
3. Sentiment (Positive, Neutral, Negative, Very Negative)
4. Brief Summary (1-2 sentences)
5. Tags (up to 5 relevant keywords)
6. Location (Extract specific branch, city, or office. If NOT found, return "Undisclosed Location")
7. Department (Extract specific department e.g., Sales, IT, HR, Logistics. If NOT found, return "General")

Respond strictly in VALID JSON format without markdown:
{{
    "category": "...",
    "priority": "...",
    "sentiment": "...",
    "summary": "...",
    "tags": ["tag1", "tag2"],
    "location": "...",
    "department": "..."
}}
"""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3
            },
            timeout=300
        )

        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Ollama API error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Ensure Ollama is running on http://localhost:11434")
        return None


# =========================
# Parse Ollama JSON Output
# =========================
def parse_ollama_response(response_text):
    """Extract and parse JSON from Ollama response."""
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response_text[start:end])
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        print("Raw response:", response_text)

    return None


# =========================
# Main Processing Logic
# =========================
def process_grievances():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        ensure_columns_exist(conn)

        cursor.execute("""
            SELECT id, transcript, language, location
            FROM grievances
            WHERE status = 'pending'
        """)
        grievances = cursor.fetchall()

        print(f"Found {len(grievances)} pending grievances")

        for idx, (gid, transcript, language, location) in enumerate(grievances, 1):
            print("\n" + "=" * 60)
            print(f"Processing {idx}/{len(grievances)} | ID: {gid}")
            print("=" * 60)

            translated_text = translate_to_english(transcript, language)
            print("✓ Translation complete")

            ollama_output = send_to_ollama(translated_text)
            if not ollama_output:
                print("✗ Ollama failed")
                continue

            analysis = parse_ollama_response(ollama_output)
            if not analysis:
                print("✗ Invalid Ollama JSON")
                continue

            print("✓ Analysis successful")
            print(f"  Category: {analysis.get('category')}")
            print(f"  Priority: {analysis.get('priority')}")
            print(f"  Sentiment: {analysis.get('sentiment')}")

            cursor.execute("""
                UPDATE grievances SET
                    translated_transcript = ?,
                    category = ?,
                    priority = ?,
                    sentiment = ?,
                    summary = ?,
                    tags = ?,
                    department = ?,
                    processed_at = ?,
                    analysis_json = ?,
                    status = 'completed'
                WHERE id = ?
            """, (
                translated_text,
                analysis.get("category", ""),
                analysis.get("priority", ""),
                analysis.get("sentiment", ""),
                analysis.get("summary", ""),
                ",".join(analysis.get("tags", [])),
                analysis.get("department", ""),
                datetime.now().isoformat(),
                json.dumps(analysis),
                gid
            ))

            conn.commit()
            print(f"✓ Grievance {gid} updated")

        conn.close()
        print("\nALL GRIEVANCES PROCESSED SUCCESSFULLY")

    except Exception as e:
        print(f"Fatal error: {e}")


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    print("Grievance Processing System")
    print("=" * 60)
    print("• Auto-translation (Deep Translator)")
    print("• Ollama-based analysis")
    print("• SQLite persistence")
    print("=" * 60)

    process_grievances()
