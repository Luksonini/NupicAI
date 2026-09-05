# NupicAI - szybkie wdrozenie jednego serwera

Ten wariant zaklada jeden proces aplikacji, jedna karte GPU i SQLite. Nie uruchamiaj
kilku workerow Uvicorn: kolejka zadan i aktywne sesje modeli sa przechowywane w
pamieci procesu.

1. Skopiuj caly katalog aplikacji do `/srv/nupicai` razem z `models/`, `tts/`,
   `translate/` i gotowym `parakeet-ui/out`.
2. Utworz uzytkownika systemowego `nupicai`, srodowisko `/srv/nupicai/.venv` i
   zainstaluj `requirements.txt`.
3. Skopiuj `.env.example` jako `.env`. Ustaw domenę, administratora, bezpieczne
   cookies, CORS i dozwolone hosty. Plik powinien miec uprawnienia `0600`.
4. Uruchom `python check_production.py --strict` i testy przed pierwszym startem.
5. Dostosuj sciezki w `nupicai.service`, zainstaluj jednostke systemd i Nginx.
6. Skonfiguruj certyfikat TLS oraz codzienny backup przez `backup_runtime.py`.

Przyklad kluczowych zmiennych produkcyjnych:

```env
NUPICAI_PRODUCTION=1
HOST=127.0.0.1
PORT=8765
NUPICAI_SECURE_COOKIES=1
NUPICAI_ALLOWED_HOSTS=nupicai.example,www.nupicai.example
WEGORZ_CORS_ORIGINS=https://nupicai.example,https://www.nupicai.example
NEXT_PUBLIC_SITE_URL=https://nupicai.example
NUPICAI_PUBLIC_URL=https://nupicai.example
RESEND_API_KEY=re_...
NUPICAI_EMAIL_FROM=NupicAI <noreply@nupicai.example>
```

Przed sprzedaza pozostaja wymagane: operator platnosci i idempotentne webhooki,
potwierdzanie e-mail, finalny przeglad prawny dokumentow,
monitoring/alerty oraz trwale znakowanie pochodzenia tresci AI.
