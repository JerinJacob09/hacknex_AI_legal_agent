"""Optional client for the Indian Kanoon API (https://api.indiankanoon.org) so real judgments can be indexed
instead of anything written from memory. Needs an API token (INDIANKANOON_TOKEN env var or the sidebar field)
and internet access; check their terms of use and pricing before bulk ingestion.

Written from the public API documentation (POST /search/?formInput=..., POST /doc/<tid>/, header
'Authorization: Token <token>'). It has NOT been run against the live service from the sandbox it was written in,
so test with one query first. Every network call goes through IndianKanoon._post, which tests replace with a fake."""
import html
import json
import re
import urllib.parse
import urllib.request

API_ROOT = "https://api.indiankanoon.org"


def html_to_text(markup):
    markup = re.sub(r"(?is)<(script|style).*?</\1>", " ", markup or "")
    markup = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</blockquote>|</pre>", "\n", markup)
    text = html.unescape(re.sub(r"<[^>]+>", " ", markup))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


class IndianKanoon:
    def __init__(self, token, timeout=30):
        if not token:
            raise ValueError("An Indian Kanoon API token is required.")
        self.token, self.timeout = token, timeout

    def _post(self, path, params=None):
        query = ("?" + urllib.parse.urlencode(params)) if params else ""
        request = urllib.request.Request(API_ROOT + path + query, data=b"", method="POST",
                                         headers={"Authorization": f"Token {self.token}", "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def search(self, query, page=0):
        data = self._post("/search/", {"formInput": query, "pagenum": page})
        return [{"tid": d.get("tid"), "title": html_to_text(d.get("title", "")), "court": d.get("docsource", ""),
                 "snippet": html_to_text(d.get("headline", ""))} for d in data.get("docs", []) if d.get("tid")]

    def fetch(self, tid):
        data = self._post(f"/doc/{tid}/")
        return {"tid": tid, "title": html_to_text(data.get("title", "")) or f"IndianKanoon {tid}",
                "text": html_to_text(data.get("doc", "")), "citation": data.get("citation") or ""}
