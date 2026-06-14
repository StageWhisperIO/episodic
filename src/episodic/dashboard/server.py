import html
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from episodic import store
from episodic.core import summary, reward
from episodic.schema import now_iso

FEEDBACK_LABELS = [
    "useful",
    "wrong",
    "too_broad",
    "too_slow",
    "needed_human_rescue",
    "accepted_as_is",
    "accepted_after_edits",
]

_STYLE = """
<style>
  body { font-family: sans-serif; margin: 2rem; color: #222; }
  h1 { font-size: 1.5rem; margin-bottom: 1rem; }
  h2 { font-size: 1.2rem; margin-top: 1.5rem; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }
  th, td { border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; }
  th { background: #f4f4f4; }
  a { color: #0066cc; }
  pre { background: #f8f8f8; border: 1px solid #ddd; padding: 1rem; overflow-x: auto; white-space: pre-wrap; }
  .badge { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px; background: #e0e0e0; font-size: 0.85rem; margin: 0.1rem; }
  .feedback-btn { margin: 0.2rem; padding: 0.3rem 0.7rem; cursor: pointer; border: 1px solid #999; background: #fff; border-radius: 3px; }
  .feedback-btn:hover { background: #f0f0f0; }
  .empty { color: #888; font-style: italic; }
</style>
"""


def escape(s):
    return html.escape(str(s) if s is not None else "")


def render_index(rows):
    title = "Episodic"
    if not rows:
        body = '<p class="empty">No episodes recorded yet.</p>'
    else:
        header = "<tr><th>ID</th><th>Intent</th><th>Outcome</th><th>Reward</th><th>Edits</th><th>Tests</th><th>Labels</th></tr>"
        row_html = []
        for r in rows:
            labels = ", ".join(escape(l) for l in (r.get("labels") or []))
            row_html.append(
                f"<tr>"
                f"<td><a href='/episode/{escape(r['id'])}'>{escape(r['id'])}</a></td>"
                f"<td>{escape(r.get('intent', ''))}</td>"
                f"<td>{escape(r.get('outcome', ''))}</td>"
                f"<td>{escape(r.get('composite_reward', ''))}</td>"
                f"<td>{escape(r.get('file_edits', ''))}</td>"
                f"<td>{escape(r.get('tests_run', ''))}</td>"
                f"<td>{labels}</td>"
                f"</tr>"
            )
        body = f"<table>{header}{''.join(row_html)}</table>"

    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>{_STYLE}</head><body><h1>{title}</h1>{body}</body></html>"


def render_episode(episode, summary_markdown):
    ep_id = escape(episode.get("id", ""))
    intent = escape(episode.get("intent", ""))
    outcome = escape(episode.get("outcome", {}).get("status", ""))

    rv = episode.get("reward_vector", {})
    reward_rows = "".join(
        f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>"
        for k, v in rv.items()
        if k != "components"
    )
    reward_table = f"<table><tr><th>Metric</th><th>Value</th></tr>{reward_rows}</table>"

    diffs = episode.get("diffs") or []
    if diffs:
        diff_header = "<tr><th>File</th><th>Status</th><th>+</th><th>-</th></tr>"
        diff_rows = "".join(
            f"<tr><td>{escape(d.get('file',''))}</td><td>{escape(d.get('status',''))}</td>"
            f"<td>{escape(d.get('additions',''))}</td><td>{escape(d.get('deletions',''))}</td></tr>"
            for d in diffs
        )
        diffs_table = f"<table>{diff_header}{diff_rows}</table>"
    else:
        diffs_table = '<p class="empty">No diffs.</p>'

    tests = episode.get("tests") or []
    if tests:
        test_header = "<tr><th>Framework</th><th>Passed</th><th>Failed</th><th>OK</th><th>Command</th></tr>"
        test_rows = "".join(
            f"<tr><td>{escape(t.get('framework',''))}</td><td>{escape(t.get('passed',''))}</td>"
            f"<td>{escape(t.get('failed',''))}</td><td>{escape(t.get('ok',''))}</td>"
            f"<td>{escape(t.get('command',''))}</td></tr>"
            for t in tests
        )
        tests_table = f"<table>{test_header}{test_rows}</table>"
    else:
        tests_table = '<p class="empty">No tests recorded.</p>'

    steps = episode.get("steps") or []
    if steps:
        step_items = "".join(
            f"<li><strong>{escape(s.get('type',''))}</strong>"
            f"{' — ' + escape(s.get('tool','')) if s.get('tool') else ''}"
            f"{': ' + escape(s.get('intent','')) if s.get('intent') else ''}"
            f"{' → ' + escape(s.get('observation','')) if s.get('observation') else ''}</li>"
            for s in steps
        )
        steps_list = f"<ol>{step_items}</ol>"
    else:
        steps_list = '<p class="empty">No steps recorded.</p>'

    btn_html = "".join(
        f"<button class='feedback-btn' onclick=\"sendFeedback('{escape(episode['id'])}','{lbl}')\">{escape(lbl)}</button>"
        for lbl in FEEDBACK_LABELS
    )

    script = """
<script>
function sendFeedback(episodeId, label) {
  fetch('/api/feedback', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({episode_id: episodeId, label: label})
  }).then(function(r) { return r.json(); }).then(function(data) {
    if (data.error) { alert('Error: ' + data.error); }
    else { alert('Feedback recorded: ' + label); location.reload(); }
  }).catch(function(e) { alert('Request failed: ' + e); });
}
</script>
"""

    body = (
        f"<h1>Episode: {ep_id}</h1>"
        f"<p><strong>Intent:</strong> {intent}</p>"
        f"<p><strong>Outcome:</strong> {outcome}</p>"
        f"<h2>Reward Vector</h2>{reward_table}"
        f"<h2>Diffs</h2>{diffs_table}"
        f"<h2>Tests</h2>{tests_table}"
        f"<h2>Steps</h2>{steps_list}"
        f"<h2>Feedback</h2>{btn_html}"
        f"<h2>Summary</h2><pre>{escape(summary_markdown)}</pre>"
        f"{script}"
    )

    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Episode {ep_id}</title>{_STYLE}</head><body>{body}</body></html>"


def apply_feedback(episode_id, label, note=None, start=None):
    episode = store.get_episode(episode_id, start)
    if episode is None:
        return {"error": "not found"}
    episode["human_feedback"].append({"ts": now_iso(), "label": label, "note": note})
    episode["reward_vector"] = reward.reward_vector(episode)
    episode["labels"] = sorted(set(episode["labels"] + [label]))
    store.save_episode(episode, start)
    return store.index_row(episode)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, code, content_type, body):
        encoded = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, html_body, code=200):
        self._send(code, "text/html; charset=utf-8", html_body)

    def _send_json(self, data, code=200):
        self._send(code, "application/json", json.dumps(data, ensure_ascii=False))

    def _not_found(self):
        self._send_html("<h1>404 Not Found</h1>", 404)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            self._send_html(render_index(store.list_episodes()))
        elif path == "/api/episodes":
            self._send_json(store.list_episodes())
        elif path.startswith("/api/episode/"):
            ep_id = path[len("/api/episode/"):]
            ep = store.get_episode(ep_id)
            if ep is None:
                self._send_json({"error": "not found"}, 404)
            else:
                self._send_json(ep)
        elif path.startswith("/episode/"):
            ep_id = path[len("/episode/"):]
            ep = store.get_episode(ep_id)
            if ep is None:
                self._not_found()
            else:
                report = summary.summarize(ep)
                md = summary.render_markdown(report)
                self._send_html(render_episode(ep, md))
        else:
            self._not_found()

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/api/feedback":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except Exception:
                self._send_json({"error": "invalid json"}, 400)
                return
            ep_id = data.get("episode_id")
            label = data.get("label")
            note = data.get("note")
            if not ep_id or not label:
                self._send_json({"error": "episode_id and label required"}, 400)
                return
            result = apply_feedback(ep_id, label, note)
            self._send_json(result)
        else:
            self._not_found()


def serve(host="127.0.0.1", port=4317, start=None):
    print(f"Episodic dashboard: http://{host}:{port}")
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
