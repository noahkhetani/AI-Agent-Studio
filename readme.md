# AI Agent Studio 🤖 Use at: https://noah-gpt.streamlit.app/

AI Agent Studio is a Streamlit application that allows you to create, customize, and manage multiple AI agents. Each agent can have its own personality, system prompt, memory settings, and OpenAI model. The app supports session memory, persistent memory, and advanced chat controls, all in a beautiful and intuitive GUI.

---

## Features

- **Multiple Agents**: Create and manage multiple AI agents with different personalities.
- **Custom System Prompt**: Define each agent's behavior and style of conversation.
- **Memory Modes**:
  - No Memory
  - Session Memory (resets when the app closes)
  - Persistent Memory (saved locally to disk)
- **Model Selection**: Choose from GPT-4, GPT-4o-mini, GPT-3.5-turbo.
- **Configurable Settings**: Temperature, max tokens, memory window.
- **Chat Interface**: Clean and colorful UI with user and AI message display.
- **Memory Management**: Clear, export, or import memory JSON files.
- **Persistent Memory Storage**: Saved in `./agent_memories/` per agent.
- **Regenerate Last Reply**: Quickly regenerate responses for the last user message.

---

## Installation

1. Clone this repository or download the `agent_studio.py` file:

```bash
git clone <repo_url>
cd <repo_folder>
```

2. Install required Python packages:

```bash
pip install streamlit openai
```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run agent_studio.py
```

2. Open the URL printed in the terminal (usually `http://localhost:8501`).
3. Enter your **OpenAI API key** in the sidebar.
4. Create a new agent or select an existing one.
5. Customize system prompt, model, and memory settings.
6. Chat with your agent in the right-hand panel.

---

## File Structure

```
agent_studio.py           # Main Streamlit app
agent_memories/           # Folder storing persistent memory JSON files
README.md                 # This file
```

---

## Security & Privacy

- Your **OpenAI API key** is used locally in the session and is **never saved or uploaded**.
- Persistent memory is stored locally on disk only in `./agent_memories/`.

---

## Future Improvements

- Token usage and cost estimation per message.
- Streaming responses for real-time chat.
- Automatic memory summarization for long-term memory management.
- Multiple AI agents interacting with each other.


---
