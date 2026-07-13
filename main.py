"""
ai agent studio - make/edit/delete a bunch of agents, each with its own system
prompt, model, temperature, max tokens and memory mode (none / session / saved
to disk). chat ui with history + memory import/export. needs openai + streamlit.
"""

import streamlit as st
import openai
import json
import os
import re
from typing import Dict, List

# helpers
def safe_filename(name: str) -> str:
    # turn an agent name into a safe filename
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    return name[:150] or "agent"

def ensure_memory_dir():
    path = os.path.join(os.getcwd(), "agent_memories")
    os.makedirs(path, exist_ok=True)
    return path

def memory_path_for(agent_id: str) -> str:
    return os.path.join(ensure_memory_dir(), f"{agent_id}_memory.json")

def load_persistent_memory(agent_id: str) -> List[Dict]:
    path = memory_path_for(agent_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_persistent_memory(agent_id: str, memory: List[Dict]):
    path = memory_path_for(agent_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def delete_persistent_memory(agent_id: str):
    path = memory_path_for(agent_id)
    if os.path.exists(path):
        os.remove(path)

def default_agent_struct(name: str):
    aid = safe_filename(name)
    return {
        "id": aid,
        "name": name,
        "system_prompt": "You are a helpful assistant.",
        "model": "gpt-3.5-turbo",
        "temperature": 0.2,
        "max_tokens": 800,
        "memory_mode": "Session Memory",  # none / session / persistent
        "memory_window": 10,  # how many prior messages to feed back in
        "messages": [],  # chat history shown in the ui
        "memory": [],    # session memory
        "persistent_loaded": False,
    }

# session state
if "agents" not in st.session_state:
    # start with one default agent
    default = default_agent_struct("Assistant")
    st.session_state.agents = {default["id"]: default}
    st.session_state.selected_agent = default["id"]

if "ui" not in st.session_state:
    st.session_state.ui = {"sidebar_open": True}

# page config + style
st.set_page_config(page_title="AI Agent Studio", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .big-title {font-size:32px; font-weight:700; color: #6b21a8; text-align:center;}
    .subtle {color: #6b7280;}
    .agent-card {background:#fbf8ff; padding:10px; border-radius:8px; margin-bottom:8px;}
    .user-msg {color:#024; background:#eef6ff; padding:8px; border-radius:6px;}
    .ai-msg {color:#023; background:#effaf1; padding:8px; border-radius:6px;}
    .small {font-size:12px; color:#6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='big-title'>AI Agent Studio</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle' style='text-align:center;margin-bottom:10px;'>Create, customize, and run multiple AI agents with memory and configurable behavior.</div>", unsafe_allow_html=True)
st.write("---")

# sidebar: api key + create agent
with st.sidebar:
    st.header("🔐 OpenAI & Agents")
    api_key = st.text_input("OpenAI API Key", type="password", help="Your API key is used only in this session and not saved by this app.")
    if api_key:
        openai.api_key = api_key  # hand it to the openai package

    st.markdown("### Agents")
    new_agent_name = st.text_input("New agent name", value="", placeholder="e.g., Research Buddy")
    new_agent_btn = st.button("➕ Create agent", key="create_agent")
    if new_agent_btn:
        name = new_agent_name.strip() or "Agent"
        agent = default_agent_struct(name)
        # keep the id unique
        uid = agent["id"]
        i = 1
        while uid in st.session_state.agents:
            uid = f"{agent['id']}_{i}"
            i += 1
        agent["id"] = uid
        st.session_state.agents[uid] = agent
        st.session_state.selected_agent = uid
        st.success(f"Created agent '{name}'")

    st.markdown("---")
    st.markdown("Select an agent to edit or chat with:")
    for aid, a in list(st.session_state.agents.items()):
        if st.button(f"Select: {a['name']}", key=f"select_{aid}"):
            st.session_state.selected_agent = aid

    st.markdown("---")
    if st.button("Export all agents (JSON)"):
        dump = json.dumps(st.session_state.agents, ensure_ascii=False, indent=2)
        st.download_button("Download agents JSON", data=dump, file_name="agents_export.json", mime="application/json")

# main layout: left = settings, right = chat
selected_id = st.session_state.get("selected_agent")
if selected_id not in st.session_state.agents:
    # fall back to whatever's first
    selected_id = next(iter(st.session_state.agents))
    st.session_state.selected_agent = selected_id

agent = st.session_state.agents[selected_id]

left, right = st.columns([0.45, 0.55])

# left: agent settings + memory
with left:
    st.subheader("Agent Settings")
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        agent_name = st.text_input("Agent name", value=agent["name"], key=f"name_{agent['id']}")
    with col2:
        if st.button("Delete agent", key=f"delete_{agent['id']}"):
            # nuke the saved memory file too if there is one
            try:
                delete_persistent_memory(agent["id"])
            except Exception:
                pass
            del st.session_state.agents[agent["id"]]
            # jump to another agent
            if st.session_state.agents:
                st.session_state.selected_agent = next(iter(st.session_state.agents))
            else:
                new = default_agent_struct("Assistant")
                st.session_state.agents[new["id"]] = new
                st.session_state.selected_agent = new["id"]
            st.experimental_rerun()

    st.text_area("System prompt (controls personality & behavior)", value=agent["system_prompt"], key=f"sys_{agent['id']}", height=140, help="E.g. 'You are a friendly, concise assistant who provides step-by-step answers.'")
    st.selectbox("Model", options=["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"], index=["gpt-4","gpt-4o-mini","gpt-3.5-turbo"].index(agent.get("model", "gpt-3.5-turbo")), key=f"model_{agent['id']}")
    st.slider("Temperature", 0.0, 1.0, value=agent.get("temperature", 0.2), step=0.05, key=f"temp_{agent['id']}")
    st.number_input("Max tokens (response)", min_value=64, max_value=4096, value=agent.get("max_tokens", 800), step=64, key=f"max_{agent['id']}")
    st.selectbox("Memory mode", ["No Memory", "Session Memory", "Persistent Memory"], index=["No Memory","Session Memory","Persistent Memory"].index(agent.get("memory_mode", "Session Memory")), key=f"memmode_{agent['id']}")
    st.number_input("Memory window (how many prior messages used)", min_value=0, max_value=200, value=agent.get("memory_window", 10), step=1, key=f"memwin_{agent['id']}")

    st.markdown("---")
    st.subheader("Memory & Storage")
    mem_mode = st.session_state.agents[selected_id]["memory_mode"] = st.session_state.get(f"memmode_{agent['id']}")
    if mem_mode == "Persistent Memory" and not agent.get("persistent_loaded"):
        # pull the saved memory in once
        loaded = load_persistent_memory(agent["id"])
        st.session_state.agents[agent["id"]]["memory"] = loaded
        st.session_state.agents[agent["id"]]["persistent_loaded"] = True

    if st.button("Clear session memory", key=f"clear_sess_{agent['id']}"):
        st.session_state.agents[agent["id"]]["memory"] = []
        st.success("Session memory cleared.")

    if st.button("Clear persistent memory", key=f"clear_pers_{agent['id']}"):
        st.session_state.agents[agent["id"]]["memory"] = []
        delete_persistent_memory(agent["id"])
        st.success("Persistent memory cleared from disk and session.")

    cola, colb = st.columns(2)
    with cola:
        st.download_button("Export memory (JSON)", data=json.dumps(agent.get("memory", []), ensure_ascii=False, indent=2), file_name=f"{agent['id']}_memory.json", mime="application/json")
    with colb:
        uploaded = st.file_uploader("Import memory (JSON)", type=["json"], key=f"upload_{agent['id']}")
        if uploaded is not None:
            try:
                loaded = json.load(uploaded)
                if isinstance(loaded, list):
                    st.session_state.agents[agent["id"]]["memory"] = loaded
                    if agent["memory_mode"] == "Persistent Memory":
                        save_persistent_memory(agent["id"], loaded)
                    st.success("Memory imported.")
                else:
                    st.error("JSON must be a list of messages [{role, content}, ...].")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    st.markdown("---")
    st.subheader("Agent Controls")
    # write the edited settings back onto the agent
    if st.button("Save settings", key=f"save_{agent['id']}"):
        st.session_state.agents[agent["id"]]["name"] = st.session_state.get(f"name_{agent['id']}", agent["name"])
        st.session_state.agents[agent["id"]]["system_prompt"] = st.session_state.get(f"sys_{agent['id']}", agent["system_prompt"])
        st.session_state.agents[agent["id"]]["model"] = st.session_state.get(f"model_{agent['id']}", agent["model"])
        st.session_state.agents[agent["id"]]["temperature"] = float(st.session_state.get(f"temp_{agent['id']}", agent["temperature"]))
        st.session_state.agents[agent["id"]]["max_tokens"] = int(st.session_state.get(f"max_{agent['id']}", agent["max_tokens"]))
        st.session_state.agents[agent["id"]]["memory_mode"] = st.session_state.get(f"memmode_{agent['id']}", agent["memory_mode"])
        st.session_state.agents[agent["id"]]["memory_window"] = int(st.session_state.get(f"memwin_{agent['id']}", agent["memory_window"]))
        st.success("Settings saved.")

# right: chat
with right:
    st.subheader(f"Chat — {agent['name']}")
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to send messages.")
    chat_container = st.container()
    if "messages" not in st.session_state.agents[agent["id"]]:
        st.session_state.agents[agent["id"]]["messages"] = []

    # newest at the bottom
    with chat_container:
        for msg in st.session_state.agents[agent["id"]]["messages"]:
            if msg.get("role") == "user":
                st.markdown(f"<div class='user-msg'><b>You:</b> {st.markdown(msg['content'], unsafe_allow_html=True) if False else msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ai-msg'><b>{agent['name']}:</b> {st.markdown(msg['content'], unsafe_allow_html=True) if False else msg['content']}</div>", unsafe_allow_html=True)

    user_text = st.text_area("Message", value="", placeholder="Type your message here...", key=f"input_{agent['id']}", height=120)
    send_col1, send_col2, send_col3 = st.columns([0.35, 0.35, 0.3])
    with send_col1:
        send_btn = st.button("Send", key=f"send_{agent['id']}")
    with send_col2:
        clear_chat = st.button("Clear chat", key=f"clearchat_{agent['id']}")
    with send_col3:
        if st.button("Regenerate last reply", key=f"regen_{agent['id']}"):
            # drop the last assistant reply and re-run the last user message
            msgs = st.session_state.agents[agent["id"]]["messages"]
            if not msgs:
                st.warning("No conversation to regenerate.")
            else:
                # find the last user message
                for i in range(len(msgs)-1, -1, -1):
                    if msgs[i]["role"] == "user":
                        last_user = msgs[i]["content"]
                        # cut everything after it
                        st.session_state.agents[agent["id"]]["messages"] = msgs[: i+1 ]
                        user_text = last_user
                        st.session_state[f"input_{agent['id']}"] = last_user
                        send_btn = True
                        break
                else:
                    st.warning("No user message found to regenerate.")

    if clear_chat:
        st.session_state.agents[agent["id"]]["messages"] = []
        st.success("Chat cleared.")

    if send_btn:
        if not api_key:
            st.error("Please provide an OpenAI API key in the sidebar to send messages.")
        elif not user_text.strip():
            st.warning("Please type a message before sending.")
        else:
            # build the message list for the api
            system = st.session_state.agents[agent["id"]]["system_prompt"]
            model = st.session_state.agents[agent["id"]]["model"]
            temperature = float(st.session_state.agents[agent["id"]]["temperature"])
            max_tokens = int(st.session_state.agents[agent["id"]]["max_tokens"])
            mem_mode = st.session_state.agents[agent["id"]]["memory_mode"]
            mem_window = int(st.session_state.agents[agent["id"]]["memory_window"])

            # system prompt first
            api_messages = []
            if system:
                api_messages.append({"role": "system", "content": system})

            # then memory, if the mode uses it (last mem_window messages)
            if mem_mode in ("Session Memory", "Persistent Memory"):
                mem_msgs = st.session_state.agents[agent["id"]].get("memory", [])
                if mem_window > 0:
                    api_messages.extend(mem_msgs[-mem_window:])
            # then the recent chat
            api_messages.extend(st.session_state.agents[agent["id"]]["messages"][-(mem_window or 0) :] )

            # and the current message
            api_messages.append({"role": "user", "content": user_text.strip()})

            try:
                with st.spinner("Sending to OpenAI..."):
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=api_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    # dig the reply out
                    reply = ""
                    if response and "choices" in response and len(response["choices"]) > 0:
                        choice = response["choices"][0]
                        if "message" in choice:
                            reply = choice["message"].get("content", "")
                        elif "text" in choice:
                            reply = choice.get("text", "")
                    reply = reply or "│ No response returned from the model."

                # add to the chat
                st.session_state.agents[agent["id"]]["messages"].append({"role": "user", "content": user_text.strip()})
                st.session_state.agents[agent["id"]]["messages"].append({"role": "assistant", "content": reply})

                # and to memory if we're using it
                if mem_mode in ("Session Memory", "Persistent Memory"):
                    st.session_state.agents[agent["id"]]["memory"].append({"role": "user", "content": user_text.strip()})
                    st.session_state.agents[agent["id"]]["memory"].append({"role": "assistant", "content": reply})
                    if mem_mode == "Persistent Memory":
                        save_persistent_memory(agent["id"], st.session_state.agents[agent["id"]]["memory"])

                st.experimental_rerun()

            except openai.error.AuthenticationError:
                st.error("Authentication failed. Check your OpenAI API key.")
            except openai.error.InvalidRequestError as e:
                st.error(f"Invalid request: {e}")
            except Exception as e:
                st.error(f"Error calling OpenAI: {e}")

# footer: tips
st.write("---")
tips_col1, tips_col2, tips_col3 = st.columns([0.33,0.33,0.34])
with tips_col1:
    st.markdown("**Quick tips**")
    st.markdown("- Use concise system prompts for predictable behavior.")
    st.markdown("- Use Persistent Memory for longer-term context between restarts.")
with tips_col2:
    st.markdown("**Shortcuts**")
    st.markdown("- Create multiple agents to experiment with different personalities.")
    st.markdown("- Export/import memories to move context between machines.")
with tips_col3:
    st.markdown("**Run notes**")
    st.markdown("- This app stores persistent memory in `./agent_memories/`.")
    st.markdown("- Keep your API key private and do not commit it to version control.")

