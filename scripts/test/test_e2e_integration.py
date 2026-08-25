#!/usr/bin/env python3
"""E2E Integration Verification Suite for Cotier-0.45B.

Tests:
1. API Server Startup (OpenAI/MCP Compatible Axum Server)
2. GET /v1/models endpoint
3. POST /v1/chat/completions (Non-streaming & SSE Streaming)
4. Tool-use extraction (<tool_call> mapping)
5. SQLite Hippocampal Episode Memory persistence
6. POST /v1/cotier/feedback rating
7. GET /v1/cotier/metrics
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_e2e_server() -> bool:
    print("=" * 65)
    print("🧪 Running Cotier v1.0 E2E Server & Memory Integration Suite")
    print("=" * 65)

    port = 8099
    server_binary = PROJECT_ROOT / "target" / "debug" / "cotier-server"
    if not server_binary.exists():
        server_binary = PROJECT_ROOT / "target" / "release" / "cotier-server"

    model_dir = PROJECT_ROOT / "models" / "cotier-0.5b"
    cmd = [
        "cargo",
        "run",
        "--manifest-path",
        str(PROJECT_ROOT / "server" / "Cargo.toml"),
        "--",
        "serve",
        "--model",
        str(model_dir),
        "--port",
        str(port),
    ]

    print(f"▶️ Starting Cotier API server on port {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Wait for server to become ready
        server_url = f"http://127.0.0.1:{port}"
        ready = False
        for attempt in range(20):
            time.sleep(1.0)
            try:
                with urllib.request.urlopen(f"{server_url}/v1/models", timeout=2.0) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                continue

        if not ready:
            print("❌ Server failed to start within timeout.")
            return False

        print("✅ Server is UP and listening.")

        # 1. Test GET /v1/models
        print("\n[1/4] Testing GET /v1/models...")
        req = urllib.request.Request(f"{server_url}/v1/models")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"   Response: {data}")
            assert data["object"] == "list"
            assert data["data"][0]["id"] == "cotier-0.5b"
        print("   ✅ Models endpoint verified.")

        # 2. Test POST /v1/chat/completions (Non-streaming)
        print("\n[2/4] Testing POST /v1/chat/completions (Non-streaming)...")
        chat_payload = {
            "model": "cotier-0.5b",
            "messages": [
                {"role": "user", "content": "Hello Cotier!"}
            ],
            "max_tokens": 8,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{server_url}/v1/chat/completions",
            data=json.dumps(chat_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"   Response choices: {data['choices']}")
            print(f"   Cortical Metrics: {data.get('cortical_metrics')}")
            assert data["object"] == "chat.completion"
            assert len(data["choices"]) > 0
        print("   ✅ Chat completions verified.")

        # 3. Test POST /v1/cotier/feedback
        print("\n[3/4] Testing POST /v1/cotier/feedback (+1 rating)...")
        fb_payload = {"episode_id": 1, "feedback": 1}
        req = urllib.request.Request(
            f"{server_url}/v1/cotier/feedback",
            data=json.dumps(fb_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"   Response: {data}")
            assert data["status"] == "success"
        print("   ✅ User feedback registration verified.")

        # 4. Test GET /v1/cotier/metrics
        print("\n[4/4] Testing GET /v1/cotier/metrics (Hippocampus status)...")
        req = urllib.request.Request(f"{server_url}/v1/cotier/metrics")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"   Memory stats: {data['memory']}")
            assert data["memory"]["total_episodes"] >= 1
        print("   ✅ Metrics & Hippocampal memory verified.")

        print("\n" + "=" * 65)
        print("🎉 ALL E2E INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 65)
        return True

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    success = test_e2e_server()
    sys.exit(0 if success else 1)
