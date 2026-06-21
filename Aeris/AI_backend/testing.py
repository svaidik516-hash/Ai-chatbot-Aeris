from flask import Flask, request, jsonify, Response, send_from_directory, redirect
from flask_cors import CORS
import os, uuid, json, re, requests, time, random
from werkzeug.utils import secure_filename
from datetime import datetime
import fitz
from PIL import Image
import pytesseract
import docx
import string
from duckduckgo_search import DDGS

app = Flask(__name__, static_folder="static")
CORS(app)

# --- CONFIGURATION ---
UPLOAD_DIR = "uploads"
HISTORY_PATH = "history.json"
FEEDBACK_PATH = "feedback.json"  # For storing user feedback
HUMANIZER_CONFIG_PATH = "humanizer_config.json" # For storing the learned prompt

os.makedirs(UPLOAD_DIR, exist_ok=True)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3"

# --- DYNAMIC PROMPT LOADING FOR AUTO-LEARNING ---
HUMANIZER_PROMPT = "" # Global variable for our prompt

def load_humanizer_prompt():
    """Loads the humanizer prompt from the config file, with a fallback default."""
    global HUMANIZER_PROMPT
    default_prompt = "Rewrite the following text using very simpler, more common words used by humans and use very basic english language also give uneven spaces after commas(,) and also remove some commas from the paragraph.IMPORTANT: Preserve the original paragraph and list structure (like '1.', '2.', etc.) exactly. Make the text sound more natural and casual:\n\n---\n\n{text}"
    
    if os.path.exists(HUMANIZER_CONFIG_PATH):
        try:
            with open(HUMANIZER_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                HUMANIZER_PROMPT = config.get("prompt", default_prompt)
                print("✅ Successfully loaded learned humanizer prompt.")
        except json.JSONDecodeError:
            HUMANIZER_PROMPT = default_prompt
            print("⚠️ Could not decode config file, using default prompt.")
    else:
        HUMANIZER_PROMPT = default_prompt
        print("ℹ️ No config file found, using default humanizer prompt.")

# Load the prompt when the app starts
load_humanizer_prompt()


# ================== SERVE FRONTEND ==================

@app.route("/")
def serve_index():
    """Serve the main frontend page"""
    return send_from_directory(app.static_folder, "Aeris.html")

@app.route("/index")
def redirect_to_ui():
    """Redirect /index to the main UI page"""
    return redirect("/")

@app.route("/<path:path>")
def serve_static(path):
    """Serve other static files (CSS, JS, images)"""
    return send_from_directory(app.static_folder, path)

# ====================================================

# --- HUMANIZER HELPER FUNCTIONS ---

# NOTE: The massive CUSTOM_WORD_DICT and replace_with_custom_dict function have been removed
# as the AI model handles this task more effectively.

def remove_symbols(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[\[\]\(\)\{\}-]', '', text)
    return text

def introduce_human_errors(text):
    lines = text.split('\n')
    processed_lines = [apply_errors(line) for line in lines]
    return '\n'.join(processed_lines)

def apply_errors(text):
    result = []
    for char in text:
        if char in ['.', ','] and random.random() < 0.65:
            continue
        result.append(char)
    processed_text = "".join(result)
    processed_text = processed_text.replace(', ', ',')
    return processed_text

def add_extra_spaces(text, probability=0.1):
    def replace(match):
        return match.group(0) + ' ' if random.random() < probability else match.group(0)
    return re.sub(r'\b \b', replace, text)

def simplify_text_with_ai(text):
    if not text or not text.strip():
        return ""
        
    # Use the dynamically loaded prompt
    prompt = HUMANIZER_PROMPT.format(text=text)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180
        )
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("response", text).strip()
    except requests.exceptions.RequestException as e:
        print(f"Error simplifying text with AI: {e}")
        return text

@app.route("/humanize", methods=["POST"])
def humanize_on_demand():
    data = request.get_json()
    original_text = data.get("text")
    if not original_text:
        return jsonify({"error": "No text provided"}), 400

    simplified_text = simplify_text_with_ai(original_text)
    cleaned_text = remove_symbols(simplified_text)
    text_with_errors = introduce_human_errors(cleaned_text)
    final_text = add_extra_spaces(text_with_errors)

    # Return both original and humanized text so frontend can send feedback
    return jsonify({
        "original_text": original_text,
        "humanized_text": final_text
    })

# ✨ NEW ENDPOINT FOR AUTO-LEARNING ✨
@app.route("/feedback", methods=["POST"])
def handle_feedback():
    """Receives feedback on a humanization task and stores it."""
    data = request.get_json()
    original_text = data.get("original_text")
    humanized_text = data.get("humanized_text")
    rating = data.get("rating") # Expecting 'good' or 'bad'

    if not all([original_text, humanized_text, rating]):
        return jsonify({"error": "Missing data for feedback"}), 400

    feedback_entry = {
        "original": original_text,
        "humanized": humanized_text,
        "rating": rating,
        "timestamp": datetime.now().isoformat()
    }

    feedback_data = []
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            try:
                feedback_data = json.load(f)
            except json.JSONDecodeError:
                pass # Start with an empty list if file is corrupt
    
    feedback_data.append(feedback_entry)

    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True, "message": "Feedback received. Thank you!"})


# --- (The rest of your backend code remains unchanged) ---

@app.route("/enhance-prompt", methods=["POST"])
def enhance_prompt():
    data = request.get_json()
    user_prompt = data.get("prompt")
    if not user_prompt or not user_prompt.strip():
        return jsonify({"error": "No prompt provided"}), 400
    enhancement_instruction = f"""You are an expert prompt engineer. Your task is to take a user's raw prompt and improve it.(you can only add 15 more extra words than the user input length)
Instructions:
1. Correct all spelling and grammatical errors.
2. Clarify any ambiguity in the user's request, making it more specific.
3. Expand the prompt by adding relevant details and context that would guide an AI towards generating a high-quality, comprehensive response, based on the user's core topic.
4. Maintain the original intent of the user's request. Do not change the topic.
5. IMPORTANT: Your output must be ONLY the enhanced prompt itself, without any introductory phrases like "Here is the enhanced prompt:" or any other conversational text.


User's raw prompt: "{user_prompt}"

Enhanced prompt:"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": enhancement_instruction,
                "stream": False,
                "options": {"temperature": 0.8}
            },
            timeout=40
        )
        response.raise_for_status()
        response_data = response.json()
        enhanced_prompt = response_data.get("response", user_prompt).strip()
        if enhanced_prompt.startswith('"') and enhanced_prompt.endswith('"'):
            enhanced_prompt = enhanced_prompt[1:-1]
        return jsonify({"enhanced_prompt": enhanced_prompt})
    except requests.exceptions.RequestException as e:
        print(f"Error enhancing prompt with AI: {e}")
        return jsonify({"error": "Failed to enhance prompt. The AI service may be unavailable."}), 503

def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f: return json.load(f)
        except: return []
    return []

def save_history(hist):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f: json.dump(hist[-100:], f, ensure_ascii=False, indent=2)

HISTORY = load_history()

def extract_text_from_file(path):
    _, ext = os.path.splitext(path); ext = ext.lower()
    try:
        if ext == ".pdf":
            with fitz.open(path) as doc: return "\n".join(page.get_text() for page in doc)
        elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
            return pytesseract.image_to_string(Image.open(path))
        elif ext == ".docx":
            doc = docx.Document(path); return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"Error extracting text from {path}: {e}")
        return ""
    return ""

def generate_prompt(user_message, conversation_turns, content_text=None):
    """
    Generates a prompt including previous conversation turns for context.
    """
    # Start with a system instruction to set the context
    prompt = "You are a helpful AI assistant. Continue the conversation below.\n\n"
    
    # Format the previous turns
    for turn in conversation_turns:
        prompt += f"User: {turn['input']}\n"
        prompt += f"Assistant: {turn['output']}\n\n"
        
    # Add any new file context
    if content_text:
        prompt += f"User has attached new files. Use the following context from them:\n{content_text}\n\n"
        
    # Add the user's latest message
    prompt += f"User: {user_message}\n"
    prompt += "Assistant:" # Prompt the model to respond as the assistant
    
    return prompt

@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message", "").strip()
    files = request.files.getlist("files")
    conversation_id = request.form.get("conversation_id")
    extracted_text = ""
    file_info = []
    for f in files:
        if f and f.filename:
            filename = secure_filename(f.filename)
            save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")
            f.save(save_path)
            extracted_text += f"\n\n--- Content from {filename} ---\n{extract_text_from_file(save_path)}"
            file_info.append(filename)

    # ✨ CHANGE 1: Find the conversation history BEFORE generating the prompt ✨
    history_turns = []
    if conversation_id:
        existing_convo = next((c for c in HISTORY if c.get("id") == conversation_id), None)
        if existing_convo:
            history_turns = existing_convo.get("turns", [])

    # ✨ CHANGE 2: Call the new, history-aware prompt function ✨
    prompt = generate_prompt(message, history_turns, extracted_text)

    def stream_response():
        full_response = ""
        nonlocal conversation_id
        try:
            with requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL, 
                    "prompt": prompt, 
                    "stream": True,
                    "options":{
                        "temperature":0.8,
                        "top_p":0.9,
                        "repeat_penalty":1.15
                    }
                },
                stream=True, timeout=180
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            if "response" in chunk:
                                token = chunk["response"]
                                full_response += token
                                for char in token:
                                    yield json.dumps({"token": char}) + "\n"
                                    time.sleep(0.01)

                            if chunk.get("done"):
                                final_text = full_response.strip()
                                yield json.dumps({"token": ""}) + "\n"

                                new_turn = {"input": message, "files": file_info, "output": final_text}
                                
                                # This part for saving remains largely the same
                                existing_convo_to_save = next((c for c in HISTORY if c.get("id") == conversation_id), None)
                                if existing_convo_to_save:
                                    existing_convo_to_save["turns"].append(new_turn)
                                    existing_convo_to_save["timestamp"] = datetime.now().isoformat()
                                else:
                                    conversation_id = str(uuid.uuid4())
                                    new_convo = {
                                        "id": conversation_id,
                                        "title": message[:50] or "Chat with attachments",
                                        "timestamp": datetime.now().isoformat(),
                                        "turns": [new_turn]
                                    }
                                    HISTORY.append(new_convo)

                                save_history(HISTORY)
                                yield json.dumps({"conversation_id": conversation_id, "history": HISTORY}) + "\n"
                                return

                        except json.JSONDecodeError:
                            print(f"Skipping malformed JSON line: {line}")
                            continue
        except Exception as e:
            yield json.dumps({"error": f"An error occurred: {e}"}) + "\n"

    return Response(stream_response(), mimetype='application/x-ndjson')

@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(HISTORY)

@app.route("/history/<conversation_id>", methods=["DELETE"])
def delete_history_item(conversation_id):
    global HISTORY
    original_length = len(HISTORY)
    HISTORY = [convo for convo in HISTORY if convo.get("id") != conversation_id]

    if len(HISTORY) < original_length:
        save_history(HISTORY)
        return jsonify({"success": True, "message": "Conversation deleted."})
    else:
        return jsonify({"success": False, "message": "Conversation not found."}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)