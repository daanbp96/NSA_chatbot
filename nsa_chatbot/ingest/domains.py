"""Editable search-domain whitelist for the discovery agent.

Persisted as a plain text file (one host per line), seeded on first read from
``config.SOURCE_DOMAINS``. Read live by ``ingest.discover`` so edits from the
Admin tab take effect without a restart.
"""

from __future__ import annotations

from urllib.parse import urlparse

from nsa_chatbot.config import SOURCE_DOMAINS, SOURCE_DOMAINS_FILE


def _normalize(raw: str) -> str:
    """A pasted URL or bare host -> a bare lowercase host (minus a leading www.)."""
    s = (raw or "").strip().lower()
    if not s:
        return ""
    if "//" not in s:
        s = "//" + s  # so urlparse treats it as a host, not a path
    parsed = urlparse(s)
    host = parsed.netloc or parsed.path
    host = host.split("/")[0].split("@")[-1].split(":")[0]  # drop path/userinfo/port
    if host.startswith("www."):
        host = host[4:]
    host = host.strip(". ")
    # Reject anything that isn't a plausible host (junk text, no TLD).
    if " " in host or "." not in host:
        return ""
    return host


def _write(domains: list[str]) -> None:
    SOURCE_DOMAINS_FILE.write_text("\n".join(domains) + "\n", encoding="utf-8")


def read_domains() -> list[str]:
    """Current whitelist. Seeds the file from SOURCE_DOMAINS on first read."""
    if not SOURCE_DOMAINS_FILE.exists():
        _write(list(SOURCE_DOMAINS))
        return list(SOURCE_DOMAINS)
    out: list[str] = []
    for line in SOURCE_DOMAINS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def add_domain(raw: str) -> list[str]:
    """Normalize and add a domain; no-op if empty or already present. Returns the list."""
    host = _normalize(raw)
    domains = read_domains()
    if host and host not in domains:
        domains.append(host)
        _write(domains)
    return domains


def remove_domain(host: str) -> list[str]:
    """Remove a domain; returns the updated list."""
    domains = [d for d in read_domains() if d != host]
    _write(domains)
    return domains
