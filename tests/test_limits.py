import asyncio

import pytest

from limits import RateLimiter, guard_client, is_quota_error


def test_sans_limite_tout_passe():
    quota = RateLimiter().new_session()
    assert all(quota.check("question") is None for _ in range(50))


def test_limite_par_session():
    limiter = RateLimiter(per_session=2)
    quota = limiter.new_session()
    assert quota.check("a") is None
    assert quota.check("b") is None
    assert "(2 questions par visite)" in quota.check("c")
    # Une autre session n'est pas affectée.
    assert limiter.new_session().check("a") is None


def test_limite_par_jour_partagee_et_remise_a_zero():
    day = {"value": "2026-10-03"}
    limiter = RateLimiter(per_day=2, today=lambda: day["value"])
    assert limiter.new_session().check("a") is None
    assert limiter.new_session().check("b") is None
    assert "aujourd'hui" in limiter.new_session().check("c")
    day["value"] = "2026-10-04"
    assert limiter.new_session().check("d") is None


def test_message_trop_long_non_compte():
    limiter = RateLimiter(per_session=1, max_chars=10)
    quota = limiter.new_session()
    assert "trop long" in quota.check("x" * 11)
    assert quota.check("court") is None


def test_from_env(monkeypatch):
    monkeypatch.setenv("GRIOT_MAX_MESSAGES_PER_SESSION", "12")
    monkeypatch.setenv("GRIOT_MAX_MESSAGES_PER_DAY", "pas un nombre")
    limiter = RateLimiter.from_env()
    assert (limiter.per_session, limiter.per_day) == (12, 0)


class _QuotaError(Exception):
    code = 429


class _FakeClient:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    async def stream_async(self, user_input, *args, **kwargs):
        self.calls += 1
        error = self.error

        async def stream():
            yield "début"
            if error:
                raise error
            yield " fin"

        return stream()


def _ask(client, text):
    async def run():
        stream = await client.stream_async(text, content="all")
        return "".join([chunk async for chunk in stream])

    return asyncio.run(run())


def test_guard_laisse_passer_puis_refuse_sans_appeler_le_llm():
    client = _FakeClient()
    guard_client(client, RateLimiter(per_session=1).new_session())
    assert _ask(client, "q1") == "début fin"
    assert "(1 questions par visite)" in _ask(client, "q2")
    assert client.calls == 1


def test_guard_traduit_une_erreur_de_quota():
    client = _FakeClient(error=_QuotaError("429 RESOURCE_EXHAUSTED"))
    guard_client(client, RateLimiter().new_session())
    answer = _ask(client, "q")
    assert answer.startswith("début")
    assert "saturé" in answer


def test_guard_ne_masque_pas_les_autres_erreurs():
    client = _FakeClient(error=RuntimeError("panne"))
    guard_client(client, RateLimiter().new_session())
    with pytest.raises(RuntimeError, match="panne"):
        _ask(client, "q")


def test_is_quota_error():
    assert is_quota_error(_QuotaError("x"))
    assert is_quota_error(Exception("429 RESOURCE_EXHAUSTED: quota"))
    assert not is_quota_error(Exception("timeout"))
