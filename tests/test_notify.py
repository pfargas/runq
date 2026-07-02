import json

import pytest

from runq import notify, store
from runq.cli import main
from runq.notify import EmailConfig, NotifyConfigError, compose_status, load_config, send


# ── a fake SMTP transport (records everything, sends nothing) ────────────────────────


class FakeSMTP:
    instances: list = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.starttls_called = False
        self.login_args = None
        self.sent = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.starttls_called = True

    def login(self, user, password):
        self.login_args = (user, password)

    def send_message(self, msg):
        self.sent.append(msg)


@pytest.fixture(autouse=True)
def _clean_env_and_fakes(monkeypatch, tmp_path):
    """Never read the user's real config or env in tests."""
    for var in ("RUNQ_EMAIL_TO", "RUNQ_SMTP_HOST", "RUNQ_SMTP_PORT", "RUNQ_SMTP_USER",
                "RUNQ_SMTP_PASSWORD", "RUNQ_EMAIL_FROM"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RUNQ_NOTIFY_CONFIG", str(tmp_path / "does-not-exist.toml"))
    FakeSMTP.instances = []


# ── config ────────────────────────────────────────────────────────────────────────


def _write_config(tmp_path, **overrides):
    lines = {"to": "me@example.com", "smtp_host": "smtp.example.com",
             "user": "me@example.com", "password": "secret", **overrides}
    body = "[email]\n" + "\n".join(
        f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}" for k, v in lines.items()
    )
    path = tmp_path / "notify.toml"
    path.write_text(body)
    return str(path)


def test_load_config_from_toml(tmp_path):
    cfg = load_config(_write_config(tmp_path, smtp_port=465), env={})
    assert cfg == EmailConfig(to="me@example.com", host="smtp.example.com", port=465,
                              user="me@example.com", password="secret",
                              sender="me@example.com")


def test_sender_fallback_chain(tmp_path):
    cfg = load_config(_write_config(tmp_path, **{"from": "other@example.com"}), env={})
    assert cfg.sender == "other@example.com"
    cfg = load_config(_write_config(tmp_path, user=""), env={})
    assert cfg.sender == "me@example.com"  # no user -> falls back to `to`


def test_env_overrides_file(tmp_path):
    path = _write_config(tmp_path)
    cfg = load_config(path, env={"RUNQ_EMAIL_TO": "cluster@example.com",
                                 "RUNQ_SMTP_PORT": "465"})
    assert cfg.to == "cluster@example.com"
    assert cfg.port == 465
    assert cfg.host == "smtp.example.com"  # rest still from the file


def test_missing_config_raises_with_hint(tmp_path):
    with pytest.raises(NotifyConfigError, match="RUNQ_EMAIL_TO"):
        load_config(str(tmp_path / "nope.toml"), env={})


# ── composition ───────────────────────────────────────────────────────────────────


def test_compose_fully_drained():
    subject, body = compose_status("outputs/runq.db", {"done": 90}, elapsed_s=7200)
    assert "fully drained" in subject
    assert "runq.db" in subject
    assert "'done': 90" in body
    assert "2.00 h" in body


def test_compose_with_failures_and_unfinished():
    subject, _ = compose_status("q.db", {"done": 5, "failed": 2})
    assert "2 FAILED" in subject
    subject, _ = compose_status("q.db", {"done": 5, "todo": 3, "running": 1})
    assert "4 unfinished" in subject  # unfinished trumps failed in the subject


# ── sending ───────────────────────────────────────────────────────────────────────


def test_send_starttls_and_login():
    cfg = EmailConfig(to="me@x.com", host="smtp.x.com", port=587,
                      user="u", password="p", sender="s@x.com")
    send("subj", "body\n", cfg, smtp_factory=FakeSMTP)
    smtp = FakeSMTP.instances[-1]
    assert smtp.starttls_called
    assert smtp.login_args == ("u", "p")
    (msg,) = smtp.sent
    assert (msg["Subject"], msg["From"], msg["To"]) == ("subj", "s@x.com", "me@x.com")
    assert msg.get_content() == "body\n"


def test_send_ssl_port_skips_starttls_and_anonymous_skips_login():
    cfg = EmailConfig(to="me@x.com", host="smtp.x.com", port=465, sender="me@x.com")
    send("subj", "body", cfg, smtp_factory=FakeSMTP)
    smtp = FakeSMTP.instances[-1]
    assert not smtp.starttls_called
    assert smtp.login_args is None


# ── CLI integration ───────────────────────────────────────────────────────────────


def _sent_via_fake(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "send", lambda subject, body, cfg, **kw: sent.append((subject, body)))
    return sent


def test_run_notify_sends_drained_email(tmp_path, toy_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNQ_EMAIL_TO", "me@example.com")
    monkeypatch.setenv("RUNQ_SMTP_HOST", "smtp.example.com")
    sent = _sent_via_fake(monkeypatch)

    db = str(tmp_path / "outputs" / "q.db")
    rc = main(["run", toy_path, "--axis", "a=1,2", "--db", db, "--serial", "--notify"])
    assert rc == 0
    (subject, body), = sent
    assert "fully drained" in subject
    assert "'done': 2" in body
    assert "notification email sent" in capsys.readouterr().out


def test_run_notify_failure_does_not_break_the_run(tmp_path, toy_path, capsys):
    # no config at all -> the sweep still succeeds, with a warning
    db = str(tmp_path / "outputs" / "q.db")
    rc = main(["run", toy_path, "--axis", "a=1", "--db", db, "--serial", "--notify"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARNING: could not send notification email" in out


def test_notify_command_reports_queue_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNQ_EMAIL_TO", "me@example.com")
    monkeypatch.setenv("RUNQ_SMTP_HOST", "smtp.example.com")
    sent = _sent_via_fake(monkeypatch)

    db = str(tmp_path / "q.db")
    conn = store.connect(db)
    store.enqueue(conn, json.dumps({"i": 0}), "i0")
    conn.close()

    assert main(["notify", "--db", db]) == 0
    (subject, _), = sent
    assert "1 unfinished" in subject
    assert "email sent" in capsys.readouterr().out


def test_notify_test_command(monkeypatch, capsys):
    monkeypatch.setenv("RUNQ_EMAIL_TO", "me@example.com")
    monkeypatch.setenv("RUNQ_SMTP_HOST", "smtp.example.com")
    sent = _sent_via_fake(monkeypatch)
    assert main(["notify", "--test"]) == 0
    assert sent[0][0] == "[runq] test email"


def test_notify_command_unconfigured_fails_loudly(capsys):
    assert main(["notify", "--test"]) == 1
    assert "could not send email" in capsys.readouterr().out
