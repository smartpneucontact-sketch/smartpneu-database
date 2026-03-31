#!/usr/bin/env python3
"""
SmartPneu Database - Tire Recognition Web App
- Camera / Upload a tire photo
- Claude Opus 4.6 vision identifies brand + model
- Checks against the JSON database
- Offers to add or edit the entry
"""

import os
import json
import base64
import functools
from datetime import datetime
from copy import deepcopy

from flask import (
    Flask, render_template, request, jsonify, session,
    redirect, url_for, send_file
)
from dotenv import load_dotenv
import anthropic
import urllib.request
import urllib.error

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smartpneu-secret-key-change-me")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_FILE = os.path.join(os.path.dirname(__file__), "brands_models.json")
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "backups")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "smartpneu")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "smartpneucontact-sketch/smartpneu-product-manager")
GITHUB_FILE_PATH = os.environ.get("GITHUB_FILE_PATH", "brands_models.json")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
data = {"brands": []}


def load_database():
    global data
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"✓ Loaded database: {len(data.get('brands', []))} brands")
        except Exception as e:
            print(f"✗ Error loading database: {e}")
            data = {"brands": []}
    else:
        print(f"! Database file not found: {DATABASE_FILE}")
        data = {"brands": []}


def save_database():
    global data
    try:
        if os.path.exists(DATABASE_FILE):
            os.makedirs(BACKUP_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = os.path.join(BACKUP_DIR, f"backup_{ts}.json")
            with open(DATABASE_FILE, "r", encoding="utf-8") as f:
                old = f.read()
            with open(backup, "w", encoding="utf-8") as f:
                f.write(old)

        with open(DATABASE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving: {e}")
        return False


def get_stats():
    brands = data.get("brands", [])
    total = sum(len(b.get("models", [])) for b in brands)
    types = {}
    for b in brands:
        for m in b.get("models", []):
            t = m.get("type", "Unknown")
            types[t] = types.get(t, 0) + 1
    return {"brands": len(brands), "models": total, "types": types}


def find_brand(name):
    """Find brand by name (case-insensitive)."""
    for b in data.get("brands", []):
        if b["name"].lower() == name.lower():
            return b
    return None


def find_model(brand_obj, model_name):
    """Find model in a brand (case-insensitive)."""
    for m in brand_obj.get("models", []):
        if m["name"].lower() == model_name.lower():
            return m
    return None


def similarity(a, b):
    """Simple similarity score (0-1) using longest common subsequence ratio."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # Check if one contains the other
    if a in b or b in a:
        return 0.9
    # Levenshtein-like: count common characters in order
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return (2.0 * lcs) / (m + n)


def find_similar_brands(query, threshold=0.45):
    """Find brands similar to query string."""
    if not query or not query.strip():
        return []
    results = []
    for b in data.get("brands", []):
        name = b["name"]
        score = similarity(query, name)
        if score >= threshold:
            results.append({
                "name": name,
                "score": round(score, 2),
                "exact": name.lower() == query.lower().strip(),
                "model_count": len(b.get("models", []))
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]


def find_similar_models(brand_name, model_query, threshold=0.45):
    """Find models in a brand similar to query string."""
    if not model_query or not model_query.strip():
        return []
    brand = find_brand(brand_name)
    if not brand:
        return []
    results = []
    for m in brand.get("models", []):
        name = m["name"]
        score = similarity(model_query, name)
        if score >= threshold:
            results.append({
                "name": name,
                "score": round(score, 2),
                "exact": name.lower() == model_query.lower().strip(),
                "model": m
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        pwd = request.form.get("password", "")
        if pwd == APP_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("index"))
        return render_template("login.html", error="Wrong password")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
@login_required
def index():
    stats = get_stats()
    brands = sorted(data.get("brands", []), key=lambda x: x["name"])
    return render_template("index.html", brands=brands, stats=stats)


# ---------------------------------------------------------------------------
# Tire recognition API  (Claude Opus 4.6 vision)
# ---------------------------------------------------------------------------
@app.route("/api/recognize", methods=["POST"])
@login_required
def recognize_tire():
    """Accept base64 image, send to Claude vision, return brand + model."""
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "Anthropic API key not configured"}), 500

    payload = request.json or {}
    image_data = payload.get("image")  # base64 string (with or without prefix)
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    # Strip data URI prefix if present
    if "," in image_data:
        header, image_data = image_data.split(",", 1)
        media_type = header.split(";")[0].split(":")[1] if ":" in header else "image/jpeg"
    else:
        media_type = "image/jpeg"

    # Build brand list for context
    brand_names = sorted([b["name"] for b in data.get("brands", [])])

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are a tire identification expert. Look at this tire photo and identify:\n"
                                "1. The brand name (manufacturer)\n"
                                "2. The model name\n"
                                "3. Any visible specs: size, speed index, load index, type (summer/winter/all-season), "
                                "3PMSF marking, runflat marking, rim protection, reinforced/XL marking.\n\n"
                                f"Known brands in our database: {', '.join(brand_names)}\n\n"
                                "Respond in this exact JSON format (no markdown, no code fences):\n"
                                '{"brand": "...", "model": "...", "type": "Ete|Hiver|4 saisons", '
                                '"speed_index": "...", "load_index": "...", "size": "...", '
                                '"3pmsf": true/false, "runflat": true/false, '
                                '"protection_jante": true/false, "renforce": true/false, '
                                '"confidence": "high|medium|low", "notes": "..."}\n\n'
                                "If you cannot read something, use null for that field. "
                                "Only return the JSON object, nothing else."
                            ),
                        },
                    ],
                }
            ],
        )

        raw = message.content[0].text.strip()
        # Try to parse JSON from the response
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)

        # Check if brand + model exist in DB
        brand_obj = find_brand(result.get("brand", ""))
        result["brand_exists"] = brand_obj is not None
        result["model_exists"] = False
        result["existing_model"] = None

        if brand_obj:
            model_obj = find_model(brand_obj, result.get("model", ""))
            if model_obj:
                result["model_exists"] = True
                result["existing_model"] = model_obj

        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({
            "error": "Could not parse AI response",
            "raw_response": raw if 'raw' in dir() else "No response"
        }), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"Anthropic API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Recognition failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Database CRUD API  (same as original, kept for the manager UI)
# ---------------------------------------------------------------------------
@app.route("/api/brands", methods=["GET"])
@login_required
def get_brands():
    brands = sorted(data.get("brands", []), key=lambda x: x["name"])
    return jsonify([{"name": b["name"], "model_count": len(b.get("models", []))} for b in brands])


@app.route("/api/brands", methods=["POST"])
@login_required
def add_brand():
    name = request.json.get("name", "").strip()
    if not name:
        return jsonify({"error": "Brand name is required"}), 400
    if find_brand(name):
        return jsonify({"error": "Brand already exists"}), 400
    data["brands"].append({"name": name, "models": []})
    save_database()
    return jsonify({"success": True, "name": name})


@app.route("/api/brands/<name>", methods=["PUT"])
@login_required
def update_brand(name):
    new_name = request.json.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "Brand name is required"}), 400
    brand = find_brand(name)
    if not brand:
        return jsonify({"error": "Brand not found"}), 404
    brand["name"] = new_name
    save_database()
    return jsonify({"success": True, "name": new_name})


@app.route("/api/brands/<name>", methods=["DELETE"])
@login_required
def delete_brand(name):
    for i, b in enumerate(data["brands"]):
        if b["name"] == name:
            del data["brands"][i]
            save_database()
            return jsonify({"success": True})
    return jsonify({"error": "Brand not found"}), 404


@app.route("/api/brands/<brand_name>/models", methods=["GET"])
@login_required
def get_models(brand_name):
    brand = find_brand(brand_name)
    if not brand:
        return jsonify({"error": "Brand not found"}), 404
    return jsonify(sorted(brand.get("models", []), key=lambda x: x["name"]))


@app.route("/api/brands/<brand_name>/models", methods=["POST"])
@login_required
def add_model(brand_name):
    md = request.json
    if not md.get("name", "").strip():
        return jsonify({"error": "Model name is required"}), 400
    brand = find_brand(brand_name)
    if not brand:
        return jsonify({"error": "Brand not found"}), 404
    if find_model(brand, md["name"]):
        return jsonify({"error": "Model already exists in this brand"}), 400
    brand["models"].append(md)
    save_database()
    return jsonify({"success": True, "model": md})


@app.route("/api/brands/<brand_name>/models/<model_name>", methods=["PUT"])
@login_required
def update_model(brand_name, model_name):
    md = request.json
    if not md.get("name", "").strip():
        return jsonify({"error": "Model name is required"}), 400
    brand = find_brand(brand_name)
    if not brand:
        return jsonify({"error": "Brand not found"}), 404
    for i, m in enumerate(brand.get("models", [])):
        if m["name"] == model_name:
            brand["models"][i] = md
            save_database()
            return jsonify({"success": True, "model": md})
    return jsonify({"error": "Model not found"}), 404


@app.route("/api/brands/<brand_name>/models/<model_name>", methods=["DELETE"])
@login_required
def delete_model(brand_name, model_name):
    brand = find_brand(brand_name)
    if not brand:
        return jsonify({"error": "Brand not found"}), 404
    for i, m in enumerate(brand.get("models", [])):
        if m["name"] == model_name:
            del brand["models"][i]
            save_database()
            return jsonify({"success": True})
    return jsonify({"error": "Model not found"}), 404


@app.route("/api/brands/<brand_name>/models/<model_name>/duplicate", methods=["POST"])
@login_required
def duplicate_model(brand_name, model_name):
    brand = find_brand(brand_name)
    if not brand:
        return jsonify({"error": "Brand not found"}), 404
    for m in brand.get("models", []):
        if m["name"] == model_name:
            new = deepcopy(m)
            new["name"] = m["name"] + " (copy)"
            brand["models"].append(new)
            save_database()
            return jsonify({"success": True, "model": new})
    return jsonify({"error": "Model not found"}), 404


@app.route("/api/stats", methods=["GET"])
@login_required
def api_stats():
    return jsonify(get_stats())


@app.route("/api/search", methods=["GET"])
@login_required
def search_models():
    q = request.args.get("q", "").lower().strip()
    if not q:
        return jsonify([])
    results = []
    for b in data["brands"]:
        for m in b.get("models", []):
            if q in m["name"].lower():
                results.append({"brand": b["name"], "model": m})
    return jsonify(results[:50])


@app.route("/api/export", methods=["GET"])
@login_required
def export_database():
    return send_file(
        DATABASE_FILE,
        as_attachment=True,
        download_name=f"tire_database_{datetime.now().strftime('%Y%m%d')}.json",
    )


@app.route("/api/import", methods=["POST"])
@login_required
def import_database():
    global data
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        content = f.read().decode("utf-8")
        new_data = json.loads(content)
        if "brands" not in new_data:
            return jsonify({"error": "Invalid database format"}), 400
        data = new_data
        save_database()
        return jsonify({"success": True, "stats": get_stats()})
    except Exception as e:
        return jsonify({"error": f"Error importing: {str(e)}"}), 400


# ---------------------------------------------------------------------------
# Similarity check API — for manual entry validation
# ---------------------------------------------------------------------------
@app.route("/api/check-brand", methods=["GET"])
@login_required
def check_brand():
    """Check if a brand name already exists or is similar to existing ones."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"matches": []})
    matches = find_similar_brands(q)
    return jsonify({"matches": matches})


@app.route("/api/check-model", methods=["GET"])
@login_required
def check_model():
    """Check if a model name already exists or is similar in a given brand."""
    brand = request.args.get("brand", "").strip()
    q = request.args.get("q", "").strip()
    if not brand or not q:
        return jsonify({"matches": []})
    matches = find_similar_models(brand, q)
    return jsonify({"matches": matches})


# ---------------------------------------------------------------------------
# GitHub Sync — push brands_models.json to product manager repo
# ---------------------------------------------------------------------------
@app.route("/api/sync", methods=["POST"])
@login_required
def sync_to_github():
    """Push the current database to the smartpneu-product-manager GitHub repo."""
    if not GITHUB_TOKEN:
        return jsonify({"error": "GitHub token not configured. Add GITHUB_TOKEN in environment variables."}), 500

    try:
        api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"

        # Step 1: Get the current file SHA (needed for update)
        get_req = urllib.request.Request(api_url, method="GET")
        get_req.add_header("Authorization", f"token {GITHUB_TOKEN}")
        get_req.add_header("Accept", "application/vnd.github.v3+json")

        current_sha = None
        try:
            with urllib.request.urlopen(get_req) as resp:
                existing = json.loads(resp.read().decode("utf-8"))
                current_sha = existing.get("sha")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise

        # Step 2: Read current database and encode as base64
        with open(DATABASE_FILE, "r", encoding="utf-8") as f:
            db_content = f.read()

        encoded = base64.b64encode(db_content.encode("utf-8")).decode("utf-8")

        # Step 3: Push to GitHub
        payload = {
            "message": f"Sync tire database — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content": encoded,
            "branch": "main",
        }
        if current_sha:
            payload["sha"] = current_sha

        put_data = json.dumps(payload).encode("utf-8")
        put_req = urllib.request.Request(api_url, data=put_data, method="PUT")
        put_req.add_header("Authorization", f"token {GITHUB_TOKEN}")
        put_req.add_header("Accept", "application/vnd.github.v3+json")
        put_req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(put_req) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        stats = get_stats()
        return jsonify({
            "success": True,
            "message": f"Synced {stats['brands']} brands and {stats['models']} models to Product Manager",
            "commit_sha": result.get("commit", {}).get("sha", ""),
        })

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return jsonify({"error": f"GitHub API error ({e.code}): {body}"}), 500
    except Exception as e:
        return jsonify({"error": f"Sync failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
load_database()

if __name__ == "__main__":
    print("=" * 50)
    print("SmartPneu Database")
    print("=" * 50)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5070)), debug=True)
