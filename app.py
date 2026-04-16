"""
Codebasics FAQs Chatbot — CSV-Powered Backend
==============================================
Knowledge base  : codebasics_faqs.csv  (prompt, response columns)
Matching engine : 4-strategy scorer (substring · token overlap · fuzzy · bigrams)
AI fallback     : Groq (free public API — llama-3.3-70b-versatile)
Profile fields  : full name · reg number · department  (hardcoded in backend only)
All original FUTO chatbot logic / features preserved.
"""

from flask import Flask, render_template, request, jsonify, session
import csv, re, random, difflib, urllib.request, json as json_lib, os, secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# ─────────────────────────────────────────────
# STUDENT PROFILE — Edit these values directly
# No UI input; details are set here in code only
# ─────────────────────────────────────────────
STUDENT_PROFILE = {
    "name":       "OBIAJUNWA FAVOUR CHIDOZIE",           # ← Change your full name here
    "reg_number": "20231412062",       # ← Change your reg number here
    "department": "Cyber Security",     # ← Change your department here
}

# ─────────────────────────────────────────────
# FREE AI FALLBACK  — Groq  (free tier, no CC)
#  Sign up at https://console.groq.com → API Keys
#  Then:  export GROQ_API_KEY=gsk_...
# ─────────────────────────────────────────────
GROQ_API_KEY = "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"

def ask_ai_fallback(question: str, profile: dict) -> str:
    name       = profile.get("name", "")
    reg_number = profile.get("reg_number", "")
    department = profile.get("department", "")

    ctx = " ".join(filter(None, [
        f"Student name: {name}."         if name       else "",
        f"Reg number: {reg_number}."     if reg_number else "",
        f"Department: {department}."     if department else "",
    ]))

    # ── Groq (free) ──────────────────────────────────────────────────────────
    if GROQ_API_KEY:
        try:
            system = (
                "You are a helpful, concise FAQ assistant for Codebasics — an online platform "
                "teaching data analytics, Python, SQL, Power BI, Excel, and machine learning. "
                f"{ctx} "
                "Answer the question clearly in under 130 words. "
                "If the question is completely unrelated to Codebasics / data / tech learning, "
                "politely say so and redirect. Use plain text with newlines where helpful."
            )
            payload = {
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 400,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": question}
                ]
            }
            body = json_lib.dumps(payload).encode()
            req  = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=body,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json_lib.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Groq error] {e}")

    # ── Google fallback (no key needed) ─────────────────────────────────────
    q_enc = question.replace(" ", "+")
    return (
        "I couldn't find a specific answer for that in my FAQ database. "
        f'<a href="https://www.google.com/search?q=Codebasics+{q_enc}" '
        f'target="_blank">Search Google: Codebasics {question}</a>'
    )


# ─────────────────────────────────────────────
# CSV KNOWLEDGE BASE
# ─────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "codebasics_faqs.csv")

def _tokens(text: str) -> list:
    """All alpha-numeric tokens, lowercased, length >= 2."""
    return re.findall(r"[a-z0-9]{2,}", text.lower())

def load_csv_kb(path: str) -> list:
    entries = []
    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            for row in csv.DictReader(f):
                prompt   = (row.get("prompt")   or "").strip()
                response = (row.get("response") or "").strip()
                if not prompt or not response:
                    continue
                combined = prompt + " " + response[:150]
                entries.append({
                    "prompt":        prompt,
                    "response":      response,
                    "tokens":        _tokens(combined),
                    "prompt_tokens": _tokens(prompt),
                    "prompt_lower":  prompt.lower(),
                })
    except FileNotFoundError:
        print(f"[WARN] CSV not found: {path}")
    return entries

KB = load_csv_kb(CSV_PATH)
print(f"[INFO] Loaded {len(KB)} FAQ entries from {CSV_PATH}")

def _short(prompt: str, n: int = 6) -> str:
    words = prompt.split()
    return " ".join(words[:n]) + ("..." if len(words) > n else "")

ALL_FAQ_TITLES = [_short(e["prompt"]) for e in KB]


# ─────────────────────────────────────────────
# 4-STRATEGY MATCHING ENGINE
# ─────────────────────────────────────────────
_STOP = {
    "i","a","an","the","is","in","it","of","to","do","my","me","for","on",
    "will","are","be","this","that","have","has","with","and","or","if",
    "at","by","was","we","you","your","can","how","what","why","not","any"
}

def _query_tokens(text: str) -> list:
    return [t for t in _tokens(text) if t not in _STOP]

def _score(query_tokens: list, entry: dict) -> float:
    qt  = query_tokens
    et  = entry["tokens"]
    ept = entry["prompt_lower"]
    score = 0.0

    qs = " ".join(qt)
    if qs and qs in ept:
        score += 25

    qt_set = set(qt)
    et_set = set(et)
    score += len(qt_set & et_set) * 7

    for t in qt:
        if len(t) >= 3 and difflib.get_close_matches(t, et, n=1, cutoff=0.82):
            score += 3

    qb = set(zip(qt, qt[1:]))
    eb = set(zip(et, et[1:]))
    score += len(qb & eb) * 5

    return score

def find_best_match(query: str) -> tuple:
    """Returns (entry_dict | None, score)."""
    qt = _query_tokens(query)
    if not qt:
        return None, 0.0
    best, best_score = None, 0.0
    for entry in KB:
        s = _score(qt, entry)
        if s > best_score:
            best_score, best = s, entry
    return best, best_score

def related_suggestions(matched: dict, n: int = 4) -> list:
    """Return n related FAQ titles by token overlap with the matched entry."""
    if not matched:
        return random.sample(ALL_FAQ_TITLES, min(n, len(ALL_FAQ_TITLES)))
    base = set(matched["tokens"])
    scored = sorted(
        [(len(base & set(e["tokens"])), _short(e["prompt"]))
         for e in KB if e is not matched],
        reverse=True
    )
    top = [t for _, t in scored if _ > 0][:n]
    if len(top) < n:
        extras = [t for t in ALL_FAQ_TITLES if t not in top]
        top += random.sample(extras, min(n - len(top), len(extras)))
    return top[:n]


# ─────────────────────────────────────────────
# CONVERSATIONAL / SOCIAL INTENTS
# ─────────────────────────────────────────────
SOCIAL_DB = {
    "greeting": {
        "patterns": ["hi","hello","hey","good morning","good afternoon",
                     "good evening","howdy","what's up","sup","hiya","greetings"],
        "full_answer": (
            "👋 <b>Hello! Welcome to the Codebasics FAQs Chatbot!</b><br><br>"
            "I'm your AI-powered guide to everything about <b>Codebasics</b> — "
            "bootcamps, courses, Power BI, Python, SQL, Excel, job assistance, and more.<br><br>"
            "Just type your question naturally and I'll find the best answer!"
        ),
        "suggestions": ["What courses does Codebasics offer?","Is there a refund policy?",
                        "Do you provide job assistance?","What is the bootcamp duration?"]
    },
    "thanks": {
        "patterns": ["thank","thanks","thank you","appreciate","helpful",
                     "good job","well done","great","perfect"],
        "full_answer": "😊 You're very welcome! Happy to help. Feel free to ask anything else about Codebasics!",
        "suggestions": ["Tell me about the bootcamp","Do you have Python courses?",
                        "How can I contact support?","Is there job assistance?"]
    },
    "goodbye": {
        "patterns": ["bye","goodbye","see you","later","take care","see ya","good night","ciao","exit"],
        "full_answer": (
            "👋 <b>Goodbye! Happy learning!</b><br><br>"
            "Come back anytime you have questions. Best of luck with your data journey! 🚀"
        ),
        "suggestions": ["Tell me about the bootcamp","What skills will I learn?","Any last tips?"]
    },
    "who_are_you": {
        "patterns": ["who are you","what are you","your name","about you",
                     "what can you do","how do you work","capabilities","features"],
        "full_answer": (
            "🤖 <b>About This Chatbot</b><br><br>"
            "I'm an AI-powered FAQ assistant for <b>Codebasics</b>.<br><br>"
            "<b>What I can do:</b><br>"
            "✅ Answer questions from the Codebasics FAQ database (CSV-powered)<br>"
            "✅ Understand typos &amp; rephrasings (fuzzy + multi-strategy matching)<br>"
            "✅ Provide personalised answers based on your student profile<br>"
            "✅ Suggest related follow-up questions<br>"
            "✅ Voice input — speak your question<br>"
            "✅ Copy any response to clipboard<br><br>"
            "📌 Ask naturally — I'll find the best match!"
        ),
        "suggestions": ["What courses does Codebasics offer?","Is there a refund policy?",
                        "Do you provide job assistance?","What skills will I learn?"]
    },
    "social": {
        "patterns": ["how are you","are you ok","what's good","you good","how do you do","how are u"],
        "full_answer": "I'm doing great, thanks for asking! 😄<br><br>Ready to help with any Codebasics questions.",
        "suggestions": ["What courses does Codebasics offer?","Do you have job assistance?",
                        "What is the bootcamp duration?","Is there a refund?"]
    },
}

def check_social(text: str):
    tl = text.lower().strip()
    for data in SOCIAL_DB.values():
        for p in data["patterns"]:
            if p in tl:
                return data["full_answer"], data.get("suggestions", [])
    return None, []


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_profile_prefix(profile: dict) -> str:
    name = profile.get("name", "").strip()
    return f"<b>{name}</b>, " if name else ""

def format_suggestions(suggestions: list) -> str:
    chips = "".join(
        f"<span class=\"suggest-chip\" onclick=\"sendFromSuggestion('{s.replace(chr(39), chr(92)+chr(39))}')\">  {s}</span>"
        for s in suggestions[:4]
    )
    return f'<div class="suggest-row">💡 You might also ask:<br>{chips}</div>'

def nl2br(text: str) -> str:
    return text.replace("\n", "<br>")

def linkify(text: str) -> str:
    return re.sub(r"(https?://[^\s<>\"']+)", r'<a href="\1" target="_blank" rel="noopener">\1</a>', text)

def fmt_response(text: str) -> str:
    return nl2br(linkify(text))


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@app.route("/get_profile", methods=["GET"])
def get_profile():
    """Expose the hardcoded student profile to the frontend (read-only)."""
    return jsonify(STUDENT_PROFILE)


@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        data    = request.get_json(force=True)
        query   = (data.get("message") or "").strip()

        # Always use the hardcoded backend profile — ignore any client-sent profile
        profile = STUDENT_PROFILE

        name_pfx = build_profile_prefix(profile)

        if not query:
            return jsonify({"reply": "Please type a question!", "suggestions": []})

        # ── SOCIAL / CONVERSATIONAL ────────────────────────────────────────────
        social_resp, social_sugs = check_social(query)
        if social_resp:
            reply = f"{name_pfx}{social_resp}<br><br>{format_suggestions(social_sugs)}"
            return jsonify({"reply": reply, "suggestions": social_sugs})

        # ── CSV KNOWLEDGE BASE MATCH ───────────────────────────────────────────
        best, score = find_best_match(query)
        MIN_CONF    = 5.0

        if best and score >= MIN_CONF:
            session["last_topic"] = best["prompt"]
            sugs   = related_suggestions(best, n=4)
            answer = fmt_response(best["response"])
            reply  = (
                f"{name_pfx}"
                f"<div class='answer-block'>{answer}</div>"
                f"<br>{format_suggestions(sugs)}"
            )
            return jsonify({"reply": reply, "suggestions": sugs})

        # ── AI FALLBACK ────────────────────────────────────────────────────────
        session["last_topic"] = None
        ai_ans  = ask_ai_fallback(query, profile)
        fb_sugs = random.sample(ALL_FAQ_TITLES, min(4, len(ALL_FAQ_TITLES)))
        reply   = (
            f"{name_pfx}That's not in my FAQ database, so I asked AI for you:<br><br>"
            f"<div class='answer-block'>{fmt_response(ai_ans)}</div>"
            f"<br>{format_suggestions(fb_sugs)}"
        )
        return jsonify({"reply": reply, "suggestions": fb_sugs})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({
            "reply": "⚠️ Something went wrong on my end. Please try again or rephrase your question.",
            "suggestions": ["What courses does Codebasics offer?","Is there a refund policy?"]
        }), 200


@app.route("/get_initial_faqs", methods=["GET"])
def get_initial_faqs():
    return jsonify({"faqs": random.sample(ALL_FAQ_TITLES, min(6, len(ALL_FAQ_TITLES)))})

@app.route("/get_new_faq", methods=["GET"])
def get_new_faq():
    return jsonify({"new_faq": random.choice(ALL_FAQ_TITLES)})


if __name__ == "__main__":
    print(f"\n  ✅ Codebasics FAQs Chatbot running!")
    print(f"  👤 Student        : {STUDENT_PROFILE['name']} | {STUDENT_PROFILE['reg_number']} | {STUDENT_PROFILE['department']}")
    print(f"  📚 Knowledge base : {len(KB)} FAQ entries from CSV")
    print(f"  🤖 AI fallback    : {'Groq (free)' if GROQ_API_KEY else 'Google search link  —  set GROQ_API_KEY to enable AI'}")
    print(f"  🌐 Open browser   : http://127.0.0.1:5000\n")
    app.run(debug=True)