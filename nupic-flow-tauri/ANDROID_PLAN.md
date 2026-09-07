# NupicAI Flow for Android - deferred implementation plan

This is the agreed scope to implement when Android work starts. It is intentionally not
part of the desktop Tauri binary.

## Stage 1: dictation and media transcription

1. Build a native Kotlin/Jetpack Compose client sharing the current NupicAI HTTP API.
2. Add contextual permission onboarding for microphone, notifications, foreground service,
   optional overlay and Android keyboard activation. Do not request unrelated permissions.
3. Provide push-to-talk, toggle and Silero VAD modes with independently configurable actions.
4. Add a custom IME keyboard so recognized text can be inserted into any active text field.
5. Add system/media audio capture through MediaProjection. Android must show its system
   confirmation for each capture session, and target applications may prohibit capture.
6. Support importing audio and call-recording files for post-processing.
7. Keep a visible recording state, local buffering, deletion controls and explicit consent.

## Stage 2: NupicAI Calls

1. Treat calling as a separate VoIP product, not as access to ordinary SIM-call audio.
2. Use WebRTC or SIP plus Android Telecom/ConnectionService integration.
3. Route both owned VoIP streams through a server-side call bridge, where VAD, diarization,
   live transcription and optional recording can operate on separate call legs.
4. Add a PSTN/SIP provider only if calls to normal phone numbers are required.
5. Before release, implement recording consent, emergency-call limitations, retention,
   abuse prevention, billing and jurisdiction-specific legal review.

## Explicit limitations

- A normal third-party Android application cannot reliably record both sides of arbitrary
  SIM, WhatsApp, Teams or other applications' calls.
- Default-dialer or call-redirection status does not grant raw two-way call audio.
- Speakerphone recording through the microphone is only a fallback and is not a production
  quality guarantee because echo cancellation may remove the remote party.
