# ai agent studio 🤖

live at: https://noah-gpt.streamlit.app/

a streamlit app for making and running a bunch of ai agents. each agent has its
own personality, system prompt, memory settings and openai model. supports
session memory, memory saved to disk, and the usual chat controls.

## what u get

- **multiple agents** - spin up as many as u want, each with its own personality
- **custom system prompt** per agent
- **memory modes**: none, session (resets on close), or persistent (saved to disk)
- pick the model (gpt-4, gpt-4o-mini, gpt-3.5-turbo)
- temperature, max tokens, memory window
- clean chat ui
- clear / export / import memory as json
- persistent memory saved per agent in `./agent_memories/`
- regenerate the last reply

## setup

```bash
git clone <repo_url>
cd <repo_folder>
pip install streamlit openai
```

## running it

```bash
streamlit run main.py
```

then open the url it prints (usually http://localhost:8501), drop ur openai api
key in the sidebar, make or pick an agent, tweak its settings, and chat on the right.

## files

```
main.py          # the streamlit app
agent_memories/  # saved memory json, one per agent
readme.md        # this
```

## privacy

ur openai key stays in the session, it's never saved or uploaded. persistent
memory only ever lives on ur own disk in `./agent_memories/`.

## stuff i might add later

- token usage / cost per message
- streaming responses
- auto-summarising long memory
- agents talking to each other
