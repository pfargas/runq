"""Email notification when a queue is drained (or on demand). Stdlib only.

Configuration lives outside the repo, in ``~/.config/runq/notify.toml``::

    [email]
    to = "you@example.com"
    smtp_host = "smtp.gmail.com"
    smtp_port = 587                 # 587 = STARTTLS (default), 465 = SSL
    user = "you@gmail.com"          # SMTP login (Gmail: use an app password)
    password = "app-password"
    from = "you@gmail.com"          # optional; defaults to user, then to

Environment variables override the file (useful on clusters): ``RUNQ_EMAIL_TO``,
``RUNQ_SMTP_HOST``, ``RUNQ_SMTP_PORT``, ``RUNQ_SMTP_USER``, ``RUNQ_SMTP_PASSWORD``,
``RUNQ_EMAIL_FROM``; ``RUNQ_NOTIFY_CONFIG`` relocates the file itself.

Verify the setup once with ``runq notify --test``.
"""

from __future__ import annotations

import os
import smtplib
import socket
import tomllib
from dataclasses import dataclass
from email.message import EmailMessage

DEFAULT_CONFIG_PATH = "~/.config/runq/notify.toml"


class NotifyConfigError(RuntimeError):
    """Email notification was requested but is not (fully) configured."""


@dataclass(frozen=True)
class EmailConfig:
    to: str
    host: str
    port: int = 587
    user: str = ""
    password: str = ""
    sender: str = ""


def load_config(path: str | None = None, env: dict | None = None) -> EmailConfig:
    """File + env-var config (env wins). Raises :class:`NotifyConfigError` if incomplete."""
    env = dict(os.environ) if env is None else env
    path = os.path.expanduser(path or env.get("RUNQ_NOTIFY_CONFIG", DEFAULT_CONFIG_PATH))
    data: dict = {}
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            data = tomllib.load(fh).get("email", {})

    def pick(key: str, envkey: str, default=""):
        return env.get(envkey) or data.get(key, default)

    to = pick("to", "RUNQ_EMAIL_TO")
    host = pick("smtp_host", "RUNQ_SMTP_HOST")
    if not to or not host:
        raise NotifyConfigError(
            f"email notification not configured: need 'to' and 'smtp_host' in the [email] "
            f"table of {path}, or RUNQ_EMAIL_TO / RUNQ_SMTP_HOST env vars"
        )
    user = pick("user", "RUNQ_SMTP_USER")
    return EmailConfig(
        to=to,
        host=host,
        port=int(pick("smtp_port", "RUNQ_SMTP_PORT", 587)),
        user=user,
        password=pick("password", "RUNQ_SMTP_PASSWORD"),
        sender=pick("from", "RUNQ_EMAIL_FROM") or user or to,
    )


def compose_status(db_path: str, counts: dict, elapsed_s: float | None = None) -> tuple[str, str]:
    """(subject, body) summarising a queue: fully drained / drained with failures / stopped."""
    host = socket.gethostname()
    unfinished = counts.get("todo", 0) + counts.get("running", 0)
    failed = counts.get("failed", 0)
    if unfinished:
        state = f"stopped with {unfinished} unfinished point(s)"
    elif failed:
        state = f"drained — {failed} FAILED"
    else:
        state = "fully drained"
    subject = f"[runq] {os.path.basename(db_path)} {state} ({host})"

    lines = [
        f"db:     {os.path.abspath(db_path)}",
        f"host:   {host}",
        f"status: {counts or 'empty'}",
    ]
    if elapsed_s is not None:
        lines.append(f"wall:   {elapsed_s / 3600:.2f} h")
    if failed:
        lines.append("\ninspect failures with:  runq failed --db " + db_path)
    return subject, "\n".join(lines) + "\n"


def send(subject: str, body: str, cfg: EmailConfig, smtp_factory=None) -> None:
    """Send one plain-text email. ``smtp_factory`` is injectable for tests."""
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, cfg.sender, cfg.to
    msg.set_content(body)
    if smtp_factory is None:
        smtp_factory = smtplib.SMTP_SSL if cfg.port == 465 else smtplib.SMTP
    with smtp_factory(cfg.host, cfg.port, timeout=30) as smtp:
        if cfg.port != 465:  # SSL connections are already encrypted; otherwise upgrade
            smtp.starttls()
        if cfg.user:
            smtp.login(cfg.user, cfg.password)
        smtp.send_message(msg)


def notify_queue(db_path: str, elapsed_s: float | None = None,
                 config_path: str | None = None) -> str:
    """Email the current status of ``db_path``'s queue. Returns the subject sent."""
    from runq import store

    cfg = load_config(config_path)
    conn = store.connect(db_path)
    counts = store.status_counts(conn)
    conn.close()
    subject, body = compose_status(db_path, counts, elapsed_s)
    send(subject, body, cfg)
    return subject


def notify_test(config_path: str | None = None) -> None:
    """Send a test email to verify the configuration (``runq notify --test``)."""
    cfg = load_config(config_path)
    send(
        "[runq] test email",
        f"runq email notification works (host {socket.gethostname()}).\n",
        cfg,
    )
