"""free-agent: минимальный ИИ-ассистент для терминала.

Подключается к любому OpenAI-совместимому API (OpenRouter, Groq, Mistral, Ollama...),
просит модель писать файлы в формате <<<FILE путь>>> ... <<<END>>> и сохраняет их
в папку workspace. История диалога хранится в JSON между запусками.
Команда /pdf превращает Markdown-файл в PDF.
"""
import json
import os
import re
import sys
import time
import pathlib

import requests

BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("AGENT_API_KEY", "")
MODEL = os.getenv("AGENT_MODEL", "qwen2.5-coder")
WORKSPACE = pathlib.Path(os.getenv("AGENT_WORKSPACE", "workspace")).resolve()
HISTORY_FILE = pathlib.Path(os.getenv("AGENT_HISTORY", str(WORKSPACE / ".history.json")))
MAX_MESSAGES = int(os.getenv("AGENT_MAX_MESSAGES", "40"))  # без system-сообщения
AUTO_YES_ENV = os.getenv("AGENT_AUTO_YES", "") == "1"
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3

FILE_RE = re.compile(r"<<<FILE (?P<path>[^\n>]+)>>>\n(?P<body>.*?)\n?<<<END>>>", re.S)

SYSTEM_PROMPT = """Ты помощник-разработчик. Отвечай по-русски, коротко и по делу.
Когда нужно создать файл (код, сайт, гайд, README), выведи его строго в формате:
<<<FILE относительный/путь.расширение>>>
содержимое файла
<<<END>>>
Не выдумывай факты о пользователе и не приписывай ему опыта, которого он не называл.
Если данных не хватает, задай один уточняющий вопрос."""

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


class AgentError(Exception):
    """Понятная пользователю ошибка (без трейсбека)."""


# ---------- файлы ----------

def parse_file_blocks(text):
    """Возвращает список (путь, содержимое) из ответа модели."""
    return [(m["path"].strip(), m["body"]) for m in FILE_RE.finditer(text)]


def safe_path(rel):
    """Путь внутри workspace; выход за его пределы запрещён."""
    path = (WORKSPACE / rel).resolve()
    if path != WORKSPACE and WORKSPACE not in path.parents:
        raise ValueError(f"Путь вне workspace: {rel}")
    return path


def write_file(rel, body):
    path = safe_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# ---------- история ----------

def system_message():
    return {"role": "system", "content": SYSTEM_PROMPT}


def trim_messages(messages, limit=None):
    """Оставляет system-сообщение и последние `limit` сообщений."""
    limit = MAX_MESSAGES if limit is None else limit
    head, rest = messages[:1], messages[1:]
    return head + rest[-limit:] if limit > 0 else head


def save_history(messages, path=None):
    """Атомарная запись истории: сначала временный файл, потом замена."""
    path = pathlib.Path(path or HISTORY_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_history(path=None):
    """Читает историю. Повреждённый файл сохраняется как .bad, диалог начинается заново."""
    path = pathlib.Path(path or HISTORY_FILE)
    fresh = [system_message()]
    if not path.exists():
        return fresh
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ok = isinstance(data, list) and all(
            isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"}
            and isinstance(m.get("content"), str) for m in data
        )
        if not ok:
            raise ValueError("неверный формат истории")
    except (OSError, ValueError) as exc:  # JSONDecodeError наследуется от ValueError
        bad = path.with_suffix(path.suffix + ".bad")
        try:
            os.replace(path, bad)
        except OSError:
            pass
        print(f"История повреждена ({exc}); старый файл: {bad}. Начинаю новый диалог.")
        return fresh
    if not data or data[0]["role"] != "system":
        data = fresh + data
    else:
        data[0] = system_message()  # всегда актуальный system-промпт
    return trim_messages(data)


# ---------- API ----------

def chat(messages):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    last_error = "неизвестная ошибка"
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, timeout=180,
                                 json={"model": MODEL, "messages": messages})
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"нет связи с API: {exc}"
        else:
            if resp.status_code in RETRY_STATUSES:
                last_error = f"HTTP {resp.status_code} (лимит или сбой сервера)"
                wait = resp.headers.get("Retry-After", "")
                if attempt < MAX_RETRIES:
                    time.sleep(float(wait) if wait.isdigit() else 2 ** attempt)
                continue
            if resp.status_code in (401, 403):
                raise AgentError("API отклонил ключ (401/403): проверьте AGENT_API_KEY и AGENT_BASE_URL.")
            if resp.status_code == 404:
                raise AgentError("404: проверьте AGENT_BASE_URL и имя модели AGENT_MODEL.")
            if not resp.ok:
                raise AgentError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            try:
                return resp.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError):
                raise AgentError(f"Неожиданный формат ответа API: {resp.text[:200]}")
        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)
    raise AgentError(f"Не удалось получить ответ после {MAX_RETRIES + 1} попыток: {last_error}")


# ---------- PDF ----------

def md_to_pdf(md_rel, pdf_rel=None):
    """Простой Markdown -> PDF (заголовки, списки, абзацы). Нужен fpdf2 и TTF-шрифт."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise AgentError("Не установлен fpdf2: pip install fpdf2")
    font = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
    if font is None:
        raise AgentError("Не найден TTF-шрифт с кириллицей (см. FONT_CANDIDATES в agent.py).")
    src = safe_path(md_rel)
    if not src.is_file():
        raise AgentError(f"Файл не найден в workspace: {md_rel}")
    out = safe_path(pdf_rel or src.with_suffix(".pdf").name)
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("U", "", font)
    pdf.set_font("U", size=11)
    for line in src.read_text(encoding="utf-8").splitlines():
        text = line.replace("**", "").replace("`", "")
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            pdf.set_font_size({1: 18, 2: 15, 3: 13}.get(level, 12))
            pdf.multi_cell(0, 9, text.lstrip("# "), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font_size(11)
        elif line.lstrip().startswith(("- ", "* ")):
            pdf.multi_cell(0, 6, "• " + text.lstrip()[2:], new_x="LMARGIN", new_y="NEXT")
        elif not line.strip():
            pdf.ln(3)
        else:
            pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(out))
    return out


# ---------- цикл ----------

def handle_reply(reply, auto_yes=False):
    for rel, body in parse_file_blocks(reply):
        if not auto_yes and input(f"Записать файл {rel}? [y/N] ").lower() != "y":
            continue
        try:
            print("Сохранён:", write_file(rel, body))
        except (ValueError, OSError) as exc:
            print("Пропущено:", exc)


def main():
    auto_yes = AUTO_YES_ENV or "--yes" in sys.argv
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    messages = load_history()
    print(f"free-agent | {MODEL} @ {BASE_URL} | workspace: {WORKSPACE}")
    print(f"История: {HISTORY_FILE} ({len(messages) - 1} сообщений). Автоподтверждение: {'да' if auto_yes else 'нет'}")
    print("Команды: /pdf файл.md, /files, /reset, /quit")
    while True:
        try:
            user = input("\nты> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            messages = [system_message()]
            save_history(messages)
            print("История очищена.")
            continue
        if user == "/files":
            print("\n".join(str(p.relative_to(WORKSPACE)) for p in WORKSPACE.rglob("*")
                            if p.is_file() and not p.name.startswith(".history")))
            continue
        if user.startswith("/pdf "):
            try:
                print("Готово:", md_to_pdf(user[5:].strip()))
            except (AgentError, ValueError, OSError) as exc:
                print("Ошибка:", exc)
            continue
        messages.append({"role": "user", "content": user})
        try:
            reply = chat(trim_messages(messages))
        except AgentError as exc:
            print("Ошибка:", exc)
            messages.pop()
            continue
        except KeyboardInterrupt:
            print("\nЗапрос прерван.")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})
        messages = trim_messages(messages)
        try:
            save_history(messages)
        except OSError as exc:
            print("Не удалось сохранить историю:", exc)
        print("\nИИ>", reply)
        handle_reply(reply, auto_yes)


if __name__ == "__main__":
    main()
