"""Garde-fous pour une démo publique : tous les visiteurs partagent le
même quota Gemini (celui de la clé du serveur).

Trois protections, réglables par variables d'environnement (0 = désactivé) :

- GRIOT_MAX_MESSAGES_PER_SESSION : messages par visite (un rechargement
  de page ouvre une nouvelle visite).
- GRIOT_MAX_MESSAGES_PER_DAY : messages par jour (UTC), tous visiteurs
  confondus. C'est le vrai plafond de consommation. Le compteur est en
  mémoire : il repart de zéro si l'application redémarre.
- GRIOT_MAX_INPUT_CHARS : longueur maximale d'un message.

Et quand Gemini répond malgré tout "quota dépassé" (erreur 429), le
visiteur voit un message clair au lieu d'une erreur technique.
"""

import os
from datetime import datetime, timezone
from typing import Any, Callable

MSG_SESSION = (
    "Vous avez atteint la limite de cette démonstration ({n} questions "
    "par visite). Merci d'avoir essayé Griot !"
)
MSG_DAY = (
    "Griot a atteint son quota de questions pour aujourd'hui (c'est une "
    "démonstration au budget limité). Revenez demain !"
)
MSG_TOO_LONG = (
    "Votre message est trop long ({n} caractères, {max} au maximum). "
    "Pouvez-vous le raccourcir ?"
)
MSG_QUOTA = (
    "Le service d'IA utilisé par Griot est momentanément saturé (quota "
    "atteint). Réessayez dans une minute."
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class RateLimiter:
    """Compteur global (par jour) partagé par toutes les sessions."""

    def __init__(
        self,
        per_session: int = 0,
        per_day: int = 0,
        max_chars: int = 2000,
        today: Callable[[], str] = _utc_day,
    ):
        self.per_session = per_session
        self.per_day = per_day
        self.max_chars = max_chars
        self._today = today
        self._day = today()
        self._count = 0

    @classmethod
    def from_env(cls) -> "RateLimiter":
        return cls(
            per_session=_env_int("GRIOT_MAX_MESSAGES_PER_SESSION", 0),
            per_day=_env_int("GRIOT_MAX_MESSAGES_PER_DAY", 0),
            max_chars=_env_int("GRIOT_MAX_INPUT_CHARS", 2000),
        )

    def _day_count(self) -> int:
        if self._today() != self._day:
            self._day, self._count = self._today(), 0
        return self._count

    def new_session(self) -> "SessionQuota":
        return SessionQuota(self)


class SessionQuota:
    """Compteur d'une session de navigateur."""

    def __init__(self, limiter: RateLimiter):
        self._limiter = limiter
        self._count = 0

    def check(self, text: str) -> str | None:
        """Renvoie un message de refus, ou None si le message est accepté
        (il est alors compté)."""
        limiter = self._limiter
        if limiter.max_chars and len(text) > limiter.max_chars:
            return MSG_TOO_LONG.format(n=len(text), max=limiter.max_chars)
        if limiter.per_session and self._count >= limiter.per_session:
            return MSG_SESSION.format(n=limiter.per_session)
        if limiter.per_day and limiter._day_count() >= limiter.per_day:
            return MSG_DAY
        self._count += 1
        limiter._count += 1
        return None


def is_quota_error(error: BaseException) -> bool:
    """Vrai pour une erreur "quota dépassé" de l'API Gemini."""
    return getattr(error, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(
        error
    )


def guard_client(client: Any, quota: SessionQuota) -> None:
    """Fait passer chaque message utilisateur par `quota` avant le LLM.

    shinychat appelle `client.stream_async(message, ...)` à chaque envoi :
    la méthode est remplacée, sur cette instance seulement, par une
    version qui refuse sans appeler le LLM quand une limite est atteinte,
    et qui traduit une erreur de quota en message lisible.
    """
    original = client.stream_async

    async def stream_async(user_input: Any = "", *args: Any, **kwargs: Any):
        refusal = quota.check(str(user_input))
        if refusal:

            async def refused():
                yield refusal

            return refused()

        async def guarded():
            try:
                stream = await original(user_input, *args, **kwargs)
                async for chunk in stream:
                    yield chunk
            except Exception as error:
                if not is_quota_error(error):
                    raise
                print(f"[QUOTA] {error}")
                yield "\n\n" + MSG_QUOTA

        return guarded()

    client.stream_async = stream_async
