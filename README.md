# mark

A macOS app that does things on your computer for you.

You type a task in plain English, like "open Safari and search for cute cats", and
mark looks at your screen, figures out the steps, and controls the mouse and keyboard
to get it done.

## Requirements

- macOS
- Python 3.13+
- An OpenAI API key
- Accessibility and Screen Recording permissions for your terminal

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put your OpenAI key in a `.env` file:

```
OPENAI_API_KEY=your-key-here
```

## Run

With a window (recommended):

```bash
python -m ui.app
```

From the command line:

```bash
python main.py "open Safari and search for cute cats"
```
