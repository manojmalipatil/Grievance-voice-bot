import sqlite3
import json
import requests
from google.cloud import translate_v2
from datetime import datetime

# Configuration
DB_PATH = r'c:\Users\ISFL-RT000265\Desktop\process\grievance.db'
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:14b"  # You can change this to your preferred model (llama2, neural-chat, etc.)

# Initialize Google Translate client (requires credentials)
try:
    translate_client = translate_v2.Client()
    TRANSLATE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Google Translate not available ({e}). Will attempt fallback.")
    TRANSLATE_AVAILABLE = False


def translate_to_english(text, language):
    """Translate text to English using Google Translate or fallback method."""
    if not text:
        return text
    
    if language.lower() == 'en' or language.lower() == 'english':
        return text
    
    try:
        if TRANSLATE_AVAILABLE:
            result = translate_client.translate_text(text)
            return result['translatedText']
    except Exception as e:
        print(f"Translation error: {e}")
    
    # If translation fails, return original text
    print(f"Warning: Could not translate from {language}. Returning original text.")
    return text


def ensure_columns_exist(conn):
    """Ensure all required columns exist in the grievances table."""
    cursor = conn.cursor()
    
    # Get existing columns
    cursor.execute("PRAGMA table_info(grievances)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    # Define new columns to add
    new_columns = {
        'translated_transcript': 'TEXT',
        'category': 'TEXT',
        'priority': 'TEXT',
        'sentiment': 'TEXT',
        'summary': 'TEXT',
        'tags': 'TEXT',
        'department': 'TEXT',
        'processed_at': 'TEXT',
        'analysis_json': 'TEXT'
    }
    
    # Add missing columns
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE grievances ADD COLUMN {col_name} {col_type}")
                print(f"Added column: {col_name}")
            except sqlite3.OperationalError as e:
                print(f"Column {col_name} already exists or error: {e}")
    
    conn.commit()


def send_to_ollama(transcript):
    """Send transcript to Ollama for analysis."""
    prompt = f"""Analyze this customer grievance and provide a structured categorization.

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

Respond strictly in VALID JSON format without markdown code blocks:
{{
    "category": "...",
    "priority": "...",
    "sentiment": "...",
    "summary": "...",
    "tags": ["tag1", "tag2"],
    "location": "...",
    "department": "..."
}}"""
    
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
            result = response.json()
            return result.get('response', '')
        else:
            print(f"Ollama API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running locally on port 11434")
        return None


def parse_ollama_response(response_text):
    """Parse JSON response from Ollama."""
    try:
        # Find JSON content in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response: {response_text}")
    
    return None


def process_grievances():
    """Main function to process pending grievances."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Ensure all columns exist
        ensure_columns_exist(conn)
        
        # Get all pending grievances
        cursor.execute("""
            SELECT id, transcript, language, location 
            FROM grievances 
            WHERE status = 'pending'
        """)
        
        pending_grievances = cursor.fetchall()
        print(f"Found {len(pending_grievances)} pending grievances to process")
        
        if not pending_grievances:
            print("No pending grievances found.")
            conn.close()
            return
        
        # Process each grievance
        for idx, (grievance_id, transcript, language, location) in enumerate(pending_grievances, 1):
            print(f"\n{'='*60}")
            print(f"Processing grievance {idx}/{len(pending_grievances)} (ID: {grievance_id})")
            print(f"{'='*60}")
            
            # Step 1: Translate to English
            print(f"Original language: {language}")
            translated_transcript = translate_to_english(transcript, language)
            print(f"Translation complete")
            
            # Step 2: Send to Ollama for analysis
            print("Sending to Ollama for analysis...")
            ollama_response = send_to_ollama(translated_transcript)
            
            if not ollama_response:
                print(f"Failed to get response from Ollama for ID {grievance_id}")
                continue
            
            # Step 3: Parse response
            print("Parsing analysis results...")
            analysis = parse_ollama_response(ollama_response)
            
            if not analysis:
                print(f"Failed to parse analysis for ID {grievance_id}")
                continue
            
            # Display results
            print("\nAnalysis Results:")
            print(f"  Category: {analysis.get('category', 'N/A')}")
            print(f"  Priority: {analysis.get('priority', 'N/A')}")
            print(f"  Sentiment: {analysis.get('sentiment', 'N/A')}")
            print(f"  Summary: {analysis.get('summary', 'N/A')}")
            print(f"  Tags: {', '.join(analysis.get('tags', []))}")
            print(f"  Location: {analysis.get('location', 'N/A')}")
            print(f"  Department: {analysis.get('department', 'N/A')}")
            
            # Step 4: Update database
            processed_at = datetime.now().isoformat()
            tags_str = ','.join(analysis.get('tags', []))
            
            cursor.execute("""
                UPDATE grievances 
                SET translated_transcript = ?,
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
                translated_transcript,
                analysis.get('category', ''),
                analysis.get('priority', ''),
                analysis.get('sentiment', ''),
                analysis.get('summary', ''),
                tags_str,
                analysis.get('department', ''),
                processed_at,
                json.dumps(analysis),
                grievance_id
            ))
            
            conn.commit()
            print(f"✓ Grievance {grievance_id} processed and updated in database")
        
        print(f"\n{'='*60}")
        print(f"Processing complete! {len(pending_grievances)} grievances processed.")
        print(f"{'='*60}")
        
        conn.close()
    
    except Exception as e:
        print(f"Error processing grievances: {e}")
        if conn:
            conn.close()


if __name__ == "__main__":
    print("Grievance Processing System")
    print("="*60)
    print("This script will:")
    print("1. Translate transcripts to English")
    print("2. Send to Ollama for analysis")
    print("3. Store results in database")
    print("="*60)
    
    process_grievances()
