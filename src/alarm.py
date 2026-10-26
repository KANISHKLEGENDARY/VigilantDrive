"""Escalating looping warning tones, with safe operation when no audio device exists."""
import math
import os
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:
    pygame = None


class Alarm:
    """Repeat a pulsing tone while a qualifying alert is active.

    Alarm level increases with a sustained alert. The visual demo continues if
    pygame or an audio output device is unavailable, and prints a clear warning.
    """
    def __init__(self, open_clear_seconds: float = 0.6,
                 escalation_seconds: tuple[float, float] = (5.0, 10.0),
                 audio_enabled: bool | None = None):
        self.open_clear_seconds = float(open_clear_seconds)
        self.escalation_seconds = escalation_seconds
        self._open_since: float | None = None
        self._started_at: float | None = None
        self._level = 0
        self._enabled = pygame is not None if audio_enabled is None else bool(audio_enabled and pygame is not None)
        self._active = False
        self._channel = None
        self._sounds = {}
        if not self._enabled:
            if audio_enabled is None and pygame is None:
                print("Audio alert disabled: install project requirements (pygame) to enable alarm sound.")
            return
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1)
            self._channel = pygame.mixer.Channel(0)
            self._sounds = {level: self._make_sound(freq, volume)
                            for level, freq, volume in ((1, 880, .18), (2, 1040, .32), (3, 1320, .5))}
        except pygame.error as exc:
            self._enabled = False
            print(f"Audio alert disabled because audio output could not start: {exc}")

    @staticmethod
    def _make_sound(frequency: int, volume: float):
        rate, tone_seconds, gap_seconds = 22050, 0.42, 0.12
        n = int(rate * tone_seconds)
        fade_n = int(rate * .015)
        envelope = np.ones(n, dtype=np.float32)
        envelope[:fade_n] = np.linspace(0, 1, fade_n)
        envelope[-fade_n:] = np.linspace(1, 0, fade_n)
        tone = np.sin(2 * math.pi * frequency * np.arange(n) / rate) * envelope * volume
        pulse = np.concatenate((tone, np.zeros(int(rate * gap_seconds), dtype=np.float32)))
        return pygame.mixer.Sound(buffer=(pulse * 32767).astype(np.int16).tobytes())

    def update(self, state: str, t: float, eye_closed: bool = False,
               yawn_active: bool = False, head_drop_active: bool = False) -> bool:
        alert_signal = state == "DROWSY" and (eye_closed or yawn_active or head_drop_active)
        if alert_signal:
            self._open_since = None
            if self._started_at is None:
                self._started_at = t
            elapsed = t - self._started_at
            level = 3 if elapsed >= self.escalation_seconds[1] else 2 if elapsed >= self.escalation_seconds[0] else 1
            newly_started = not self._active
            if self._enabled and (level != self._level or newly_started):
                self._channel.play(self._sounds[level], loops=-1)
            self._level = level
            self._active = True
            return newly_started
        if self._active:
            if self._open_since is None:
                self._open_since = t
            if t - self._open_since >= self.open_clear_seconds:
                if self._enabled:
                    self._channel.stop()
                self._started_at = None
                self._open_since = None
                self._level = 0
                self._active = False
        else:
            self._started_at = None
            self._open_since = None
            self._level = 0
        return False

    def close(self) -> None:
        if self._enabled:
            self._channel.stop()
            pygame.mixer.quit()
        self._active = False
