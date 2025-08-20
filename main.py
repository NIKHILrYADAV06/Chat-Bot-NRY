# app.py
from flask import Flask, render_template, request, jsonify
import google.genai as genai
from google.genai import types
import os
from dotenv import load_dotenv
from datetime import datetime
import re

load_dotenv()

app = Flask(__name__)

# ---------- API KEY ----------
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("[WARN] OPENAI_API_KEY is not set. Set it in your environment for the app to work.")

client = genai.Client(api_key=API_KEY)

# ---------- MODELS ----------
TEXT_MODEL = "gemini-2.5-flash"
VISION_MODEL = "gemini-2.5-pro"

# ---------- ENHANCED CHAT HISTORY WITH PERSISTENCE ----------
chat_sessions = {}  # {session_id: {history: [types.Content], created_at: str, title: str, pinned: bool}}
current_session_id = None

def now_time():
    return datetime.now().strftime("%H:%M")

def new_session_title():
    return f"New Chat {len(chat_sessions) + 1}"

def get_or_create_default_session():
    global current_session_id
    if not chat_sessions:
        sid = create_session_object()
        current_session_id = sid
    return chat_sessions[current_session_id]

def slug_from_text(text: str, limit=40):
    text = re.sub(r"\s+", " ", text.strip())
    text = text[:limit]
    return text if text else None

def create_session_object():
    session_id = f"session_{len(chat_sessions) + 1}_{int(datetime.now().timestamp())}"
    chat_sessions[session_id] = {
        "history": [],
        "created_at": datetime.now().isoformat(),
        "title": new_session_title(),
        "pinned": False,
    }
    return session_id

@app.route("/")
def index():
    # Make sure at least one session exists
    get_or_create_default_session()
    return render_template("index.html")

# ---------- SESSIONS API ----------
@app.get("/api/sessions")
def get_sessions():
    # Build list and sort: pinned first, then newest created
    items = []
    for sid, data in chat_sessions.items():
        msg_count = sum(1 for c in data["history"] if getattr(c, "role", "") == "user")
        items.append({
            "id": sid,
            "title": data["title"],
            "created_at": data["created_at"],
            "message_count": msg_count,
            "pinned": data.get("pinned", False)
        })
    items.sort(key=lambda x: ((not x["pinned"]), x["created_at"]), reverse=True)
    return jsonify({"sessions": items, "current": current_session_id})

@app.get("/api/sessions/<session_id>")
def get_session(session_id):
    if session_id not in chat_sessions:
        return jsonify({"error": "Session not found"}), 404

    session = chat_sessions[session_id]
    messages = []
    for content in session["history"]:
        message_data = {
            "role": content.role,
            "timestamp": now_time(),
            "parts": []
        }
        for part in content.parts:
            if hasattr(part, "text") and part.text:
                message_data["parts"].append({"type": "text", "content": part.text})
            elif hasattr(part, "inline_data"):
                message_data["parts"].append({"type": "image", "content": "image_data"})
        messages.append(message_data)

    return jsonify({
        "session": {
            "id": session_id,
            "title": session["title"],
            "created_at": session["created_at"],
            "messages": messages
        }
    })

@app.post("/api/sessions")
def create_session():
    global current_session_id
    sid = create_session_object()
    current_session_id = sid
    return jsonify({"session_id": sid, "title": chat_sessions[sid]["title"]})

@app.put("/api/sessions/<session_id>")
def update_session(session_id):
    if session_id not in chat_sessions:
        return jsonify({"error": "Session not found"}), 404
    data = request.get_json(silent=True) or {}
    if "title" in data:
        chat_sessions[session_id]["title"] = (data["title"] or "").strip() or chat_sessions[session_id]["title"]
    if "pinned" in data:
        chat_sessions[session_id]["pinned"] = bool(data["pinned"])
    return jsonify({"ok": True})

@app.delete("/api/sessions/<session_id>")
def delete_session(session_id):
    global current_session_id
    if session_id not in chat_sessions:
        return jsonify({"error": "Session not found"}), 404
    del chat_sessions[session_id]
    if current_session_id == session_id:
        current_session_id = next(iter(chat_sessions.keys()), None)
        if current_session_id is None:
            current_session_id = create_session_object()
    return jsonify({"ok": True, "current": current_session_id})

@app.post("/api/switch-session/<session_id>")
def switch_session(session_id):
    global current_session_id
    if session_id not in chat_sessions:
        return jsonify({"error": "Session not found"}), 404
    current_session_id = session_id
    return jsonify({"ok": True, "session_id": session_id})

@app.post("/reset")
def reset():
    # clear only the current session history
    sess = get_or_create_default_session()
    sess["history"].clear()
    return jsonify({"ok": True})

# ---------- CHAT ----------
@app.post("/chat")
def chat():
    """
    Accepts either JSON or multipart form.
    - JSON: { "message": "text" }
    - Multipart: files under 'images' + 'message'
    """
    user_text = ""
    image_parts = []

    # Detect multipart (image upload) vs JSON
    if request.content_type and request.content_type.startswith("multipart/"):
        user_text = (request.form.get("message") or "").strip()
        files = request.files.getlist("images")
        for f in files:
            if not f:
                continue
            b = f.read()
            mime = f.mimetype or "application/octet-stream"
            name = (f.filename or "").lower()
            if mime == "application/octet-stream":
                if name.endswith(".png"):
                    mime = "image/png"
                elif name.endswith(".jpg") or name.endswith(".jpeg"):
                    mime = "image/jpeg"
                elif name.endswith(".webp"):
                    mime = "image/webp"
                else:
                    mime = "image/*"
            try:
                image_parts.append(types.Part.from_bytes(data=b, mime_type=mime))
            except Exception as e:
                print("[WARN] Could not make image Part:", e)
    else:
        data = request.get_json(silent=True) or {}
        user_text = (data.get("message") or "").strip()

    if not user_text and not image_parts:
        return jsonify({"reply": "Please type or upload something.", "time": now_time()}), 200

    # Ensure we have a current session
    sess = get_or_create_default_session()
    history = sess["history"]

    # Add user content
    user_parts = []
    if user_text:
        user_parts.append(types.Part.from_text(text=user_text))
    user_parts.extend(image_parts)

    history.append(types.Content(role="user", parts=user_parts))

    # Auto-title the session from first user text (once)
    if sess["title"].startswith("New Chat") and user_text:
        maybe = slug_from_text(user_text, 40)
        if maybe:
            sess["title"] = maybe

    # Choose model
    model_name = VISION_MODEL if image_parts else TEXT_MODEL
    config = types.GenerateContentConfig()

    reply_text = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=history,
            config=config,
        ):
            if chunk.text:
                reply_text += chunk.text
    except Exception as e:
        reply_text = f"Model error: {e}"

    history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))

    return jsonify({"reply": reply_text, "time": now_time()}), 200

if __name__ == "__main__":
    # In production set debug=False
    app.run(host="0.0.0.0", port=3000, debug=True)

