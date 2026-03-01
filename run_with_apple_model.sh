#!/bin/bash

# Configuration
SWIFT_DIR="LocalAppleOpenAIServer"
VENV_DIR=".venv"

# Function to find a free port on macOS
get_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

# 1. Setup Virtual Environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "🛠 Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install rich exa_py voyageai anthropic openai pydantic pydantic-settings requests websockets termcolor dataclasses-json python-dotenv pymysql
else
    source "$VENV_DIR/bin/activate"
fi

# 2. Select a free port
export PORT=$(get_free_port)
echo "🚀 Selected free port: $PORT"

# 3. Kill any existing instances
pkill -9 apple-bridge 2>/dev/null
pkill -9 App 2>/dev/null
pkill -9 LocalAppleOpenAIServer 2>/dev/null

# 4. Start the Swift Server in the background
echo "📦 Starting Local Apple OpenAI Server..."
if [ -f "./apple-bridge" ]; then
    nohup ./apple-bridge serve --port "$PORT" > server.log 2>&1 &
else
    cd "$SWIFT_DIR"
    swift build > /dev/null 2>&1
    nohup .build/debug/App serve --port "$PORT" > server.log 2>&1 &
    cd ..
fi
SERVER_PID=$!

# 5. Wait for the server to be ready
echo "⏳ Waiting for server to start on http://127.0.0.1:$PORT..."
MAX_RETRIES=30
COUNT=0
while ! curl -s "http://127.0.0.1:$PORT" > /dev/null; do
    sleep 1
    COUNT=$((COUNT + 1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Server failed to start. Check $SWIFT_DIR/server.log"
        tail -n 20 "$SWIFT_DIR/server.log"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done
echo "✅ Server is up and running with REAL Apple Foundation Model!"

# 6. Set OpenPlanter environment variables
export OPENPLANTER_PROVIDER="apple"
export OPENPLANTER_APPLE_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENPLANTER_APPLE_API_KEY="local-apple"
export OPENPLANTER_MODEL="apple-foundation-model"
export OPENPLANTER_RECURSIVE="false" # Local model is better in flat mode
export OPENPLANTER_TIMEOUT="600"     # Give the local model more time

echo "🤖 Launching OpenPlanter..."
echo "------------------------------------------------------------"

# 7. Run OpenPlanter
python3 -m agent "$@"

# 8. Cleanup after OpenPlanter exits
echo "------------------------------------------------------------"
echo "🛑 Shutting down local server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null
