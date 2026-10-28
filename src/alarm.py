"""Play the sound selected by the visible drowsiness state without blocking video."""
import math
import os
import queue
import threading
import time
import numpy as np
from src.config import (ALERT_POLICIES, ALERT_PRIORITY, AUDIO_FADE_SECONDS,
                        AUDIO_SAMPLE_RATE, AUDIO_TONE_SECONDS,
                        MINIMUM_ALARM_SECONDS, SAFE_CLEAR_SECONDS, SAFE_GRACE_SECONDS)

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:
    pygame = None


class Alarm:
    """Map one state to sound and hold the alert until the driver is safe."""
    def __init__(self, open_clear_seconds: float = SAFE_CLEAR_SECONDS,
                 minimum_alarm_seconds: float = MINIMUM_ALARM_SECONDS,
                 safe_grace_seconds: float = SAFE_GRACE_SECONDS,
                 audio_enabled: bool | None = None):
        self.open_clear_seconds = float(open_clear_seconds)
        self.minimum_alarm_seconds = float(minimum_alarm_seconds)
        self.safe_grace_seconds = float(safe_grace_seconds)
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.audio_channels = 1
        self._enabled = pygame is not None if audio_enabled is None else bool(audio_enabled and pygame is not None)
        self._requested_state = "NORMAL"
        self._actual_state = "NORMAL"
        self._started_at: float | None = None
        self._safe_since: float | None = None
        self._unsafe_since: float | None = None
        self._commands: queue.Queue = queue.Queue()
        self._thread = None
        self._channel = None
        self._sounds = {}
        if not self._enabled:
            if audio_enabled is None and pygame is None:
                print("Audio alert disabled: install project requirements (pygame) to enable alarm sound.")
            return
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1)
            mixer_settings = pygame.mixer.get_init()
            if mixer_settings is None:
                raise pygame.error("Audio mixer did not report its active format")
            self.sample_rate, _, self.audio_channels = mixer_settings
            print(f"Audio output: {self.sample_rate} Hz, {self.audio_channels} channel(s)")
            self._channel = pygame.mixer.Channel(0)
            self._sounds = {state: self._make_sound(policy["frequency"], policy["volume"],
                                                    self.sample_rate, self.audio_channels)
                            for state, policy in ALERT_POLICIES.items() if policy["sound"] != "silent"}
            self._thread = threading.Thread(target=self._play_loop, daemon=True)
            self._thread.start()
        except pygame.error as exc:
            self._enabled = False
            print(f"Audio alert disabled because audio output could not start: {exc}")

    @staticmethod
    def _make_sound(frequency: int, volume: float, rate: int = AUDIO_SAMPLE_RATE,
                    channels: int = 1):
        tone_seconds = AUDIO_TONE_SECONDS
        n = int(rate * tone_seconds)
        fade_n = int(rate * AUDIO_FADE_SECONDS)
        envelope = np.ones(n, dtype=np.float32)
        envelope[:fade_n] = np.linspace(0, 1, fade_n)
        envelope[-fade_n:] = np.linspace(1, 0, fade_n)
        tone = np.sin(2 * math.pi * frequency * np.arange(n) / rate) * envelope * volume
        samples = (tone * 32767).astype(np.int16)
        if channels > 1:
            samples = np.repeat(samples[:, None], channels, axis=1)
        return pygame.mixer.Sound(buffer=samples.tobytes())

    @staticmethod
    def _tone_allowed(policy: dict, played: int) -> bool:
        """Return whether this state has more scheduled beeps remaining."""
        maximum = policy.get("max_beeps")
        return maximum is None or played < maximum

    def _play_loop(self) -> None:
        """Play short tones on a worker thread so audio never stalls frames."""
        state = "NORMAL"
        next_tone = 0.0
        beeps_played = 0
        while True:
            try:
                new_state = self._commands.get(timeout=0.05)
                if new_state is None:
                    break
                state = new_state
                next_tone = 0.0
                beeps_played = 0
                if state == "NORMAL":
                    self._channel.stop()
            except queue.Empty:
                pass
            policy = ALERT_POLICIES[state]
            now = time.monotonic()
            if (policy["sound"] != "silent" and now >= next_tone
                    and self._tone_allowed(policy, beeps_played)):
                self._channel.play(self._sounds[state])
                beeps_played += 1
                next_tone = now + policy["repeat_seconds"]

    def update(self, state: str, t: float, safe_to_clear: bool = False) -> str:
        """Return the actual alarm state; this same value is shown on screen."""
        if state not in ALERT_POLICIES:
            raise ValueError(f"Unknown alert state: {state}")
        if state != "NORMAL" and ALERT_PRIORITY[state] >= ALERT_PRIORITY[self._actual_state]:
            self._safe_since = None
            self._unsafe_since = None
            if self._started_at is None:
                self._started_at = t
            self._set_state(state)
        elif state == self._actual_state:
            self._safe_since = None
            self._unsafe_since = None
        elif self._actual_state != "NORMAL":
            if safe_to_clear:
                self._unsafe_since = None
                self._safe_since = self._safe_since if self._safe_since is not None else t
            elif self._safe_since is not None:
                self._unsafe_since = self._unsafe_since if self._unsafe_since is not None else t
                if t - self._unsafe_since > self.safe_grace_seconds:
                    self._safe_since = None
                    self._unsafe_since = None
            minimum_elapsed = self._started_at is not None and t - self._started_at >= self.minimum_alarm_seconds
            safe_elapsed = self._safe_since is not None and t - self._safe_since >= self.open_clear_seconds
            if minimum_elapsed and safe_elapsed:
                self._set_state(state)
                self._started_at = None if state == "NORMAL" else t
                self._safe_since = None
        return self._actual_state

    def _set_state(self, state: str) -> None:
        """Update the shared alarm state and notify its sound worker."""
        if state == self._actual_state:
            return
        self._actual_state = self._requested_state = state
        if self._enabled:
            while True:
                try:
                    self._commands.get_nowait()
                except queue.Empty:
                    break
            if state == "NORMAL":
                self._channel.stop()
            self._commands.put(state)

    def force_normal(self) -> None:
        """Stop audio immediately when the driver restarts calibration."""
        self._started_at = None
        self._safe_since = None
        self._unsafe_since = None
        if self._actual_state != "NORMAL":
            self._set_state("NORMAL")
        elif self._enabled:
            self._channel.stop()
            while True:
                try:
                    self._commands.get_nowait()
                except queue.Empty:
                    break
            self._commands.put("NORMAL")

    def close(self) -> None:
        if self._enabled:
            self._commands.put(None)
            self._thread.join(timeout=1.0)
            pygame.mixer.quit()
        self._actual_state = "NORMAL"
