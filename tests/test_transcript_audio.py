"""Buffered STT engines get the full audio; extraction can't bypass the gate.

Drives ``_finish_session`` end to end with fakes, so the gate, extraction
and the buffered engine run for real.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np

from murdock.config import Settings
from murdock.core.context import AppContext
from murdock.core.db import open_db
from murdock.core.recognition_log import RecognitionLog
from murdock.core.speaker_store import VerificationResult
from wyoming_murdock.handler import MurdockHandler

_RATE = 16000

# Each voice is a constant sample value.
CHIME = 500
JONAS = 2000      # enrolled
STRANGER = 3000   # not enrolled


def _tone(value: int, seconds: float) -> np.ndarray:
    return np.full(int(seconds * _RATE), value, dtype=np.int16)


def _gap(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * _RATE), dtype=np.int16)


def _pcm(*parts: np.ndarray) -> bytes:
    return np.concatenate(parts).astype("<i2").tobytes()


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2")


class _RunVAD:
    """Speech segments = runs of the same non-zero sample value."""

    def analyze_pcm(self, pcm: bytes):
        s = _samples(pcm)
        segments, start = [], None
        for i in range(len(s) + 1):
            v = s[i] if i < len(s) else 0
            if start is not None and (v == 0 or v != s[start]):
                segments.append((start / _RATE, i / _RATE))
                start = None
            if start is None and v != 0:
                start = i
        return SimpleNamespace(
            segments=segments,
            speech_seconds=sum(e - st for st, e in segments),
        )


class _ShareEmbedder:
    """The 'embedding' is the share of voiced samples that are Jonas."""

    def embed_pcm(self, pcm: bytes) -> np.ndarray:
        s = _samples(pcm)
        voiced = s[s != 0]
        if voiced.size == 0:
            raise ValueError("no speech")
        return np.array([float(np.mean(voiced == JONAS))])


class _Speakers:
    def list_speakers(self):
        return [SimpleNamespace(id=1, name="Jonas")]

    def get_speaker(self, speaker_id):
        return SimpleNamespace(id=1, name="Jonas", role="admin")

    def verify_embedding(self, embedding, threshold=0.3, **_kw):
        distance = 1.0 - float(embedding[0])
        hit = distance <= threshold
        return VerificationResult(
            is_match=hit,
            matched_speaker="Jonas" if hit else None,
            matched_speaker_id=1 if hit else None,
            distance=distance,
            threshold=threshold,
            all_distances={"Jonas": distance},
        )


class _Engine:
    """A buffered cloud engine that records what it was sent."""

    def __init__(self):
        self.audio: list[bytes] = []
        self.last_timing = {}

    async def transcribe(self, audio, **_kw):
        self.audio.append(audio)
        return "Licht an"


def _setup(tmp_path, *, require_match=False, extraction=True):
    db = open_db(tmp_path / "m.db")
    ctx = AppContext(
        settings=Settings(), db=db, embedder=_ShareEmbedder(), vad=_RunVAD(),
        speakers=_Speakers(), unknown=None, ha=None, mqtt=None,
        recognition=RecognitionLog(db),
    )
    main = ctx.stt_services.create("OpenAI", "openai", {"api_key": "k", "model": "m"})
    engine = _Engine()
    ctx.build_stt_backend = lambda svc: engine if svc and svc.id == main.id else None
    ctx.set_enable_stt_prep(False)        # the engine sees the raw buffer
    ctx.set_min_liveness_score(0.0)       # no spectral heuristic on tones
    ctx.set_skip_leading_seconds(1.0)
    ctx.set_require_match(require_match)
    ctx.set_enable_extraction(extraction)
    ctx.set_unknown_logging(False)
    ctx.set_auto_enroll(False)
    ctx.detect_whisper = lambda audio: None

    async def _no_media(*_a, **_kw):
        return 0.0

    ctx.compute_media_tightening = _no_media
    published: list[dict] = []
    ctx.publish_recognition = lambda **kw: published.append(kw)

    async def _publish_now(**kw):
        published.append(kw)

    ctx.publish_recognition_now = _publish_now

    handler = MurdockHandler(
        SimpleNamespace(upstream_uri="tcp://unused:1"), ctx,
        SimpleNamespace(), SimpleNamespace(),
    )
    sent: list[str] = []

    async def _send(text, label=""):
        sent.append(text)

    handler._send_transcript_to_satellite = _send
    return ctx, handler, engine, sent, published


def _run(handler, pcm: bytes) -> None:
    async def go():
        handler._reset_session_state()
        assert handler._is_voxtral, "the test needs the buffered path"
        handler._audio_buffer = bytearray(pcm)
        handler._stream_start = time.monotonic() - len(pcm) / (2 * _RATE)
        await handler._finish_session()
        await asyncio.sleep(0.05)  # let fire-and-forget log writes land
    asyncio.run(go())


def test_the_engine_hears_the_first_second(tmp_path):
    ctx, handler, engine, sent, _ = _setup(tmp_path, extraction=False)
    pcm = _pcm(_tone(CHIME, 0.3), _tone(JONAS, 0.7), _tone(JONAS, 2.0))
    _run(handler, pcm)
    assert sent == ["Licht an"]
    assert len(engine.audio) == 1
    assert engine.audio[0] == pcm, "the engine got a trimmed copy"


def test_short_regions_dropped_by_extraction_still_reach_the_engine(tmp_path):
    ctx, handler, engine, sent, _ = _setup(tmp_path)
    # The 0.4 s word is too short to score, so extraction drops it.
    pcm = _pcm(
        _tone(CHIME, 1.0), _tone(JONAS, 2.0), _gap(0.1), _tone(JONAS, 0.4),
    )
    _run(handler, pcm)
    assert sent == ["Licht an"]
    assert engine.audio == [pcm], "the engine lost the short region"


def test_one_enrolled_word_does_not_carry_a_strangers_command(tmp_path):
    ctx, handler, engine, sent, published = _setup(tmp_path, require_match=True)
    pcm = _pcm(
        _tone(CHIME, 1.0),
        _tone(JONAS, 0.7),          # "ja"
        _gap(0.1),
        _tone(STRANGER, 2.3),       # "öffne die Haustür"
    )
    _run(handler, pcm)
    assert sent == [""], "an unknown voice's command passed the gate"
    assert engine.audio == [], "a blocked utterance was transcribed"
    assert not any(p.get("is_known") for p in published), (
        "the agent was told an enrolled speaker was talking"
    )


def test_one_enrolled_word_does_not_lend_its_name_without_the_gate(tmp_path):
    ctx, handler, engine, sent, published = _setup(tmp_path, require_match=False)
    pcm = _pcm(
        _tone(CHIME, 1.0), _tone(JONAS, 0.7), _gap(0.1), _tone(STRANGER, 2.3),
    )
    _run(handler, pcm)
    assert sent == ["Licht an"]
    assert engine.audio == [pcm]
    assert published and not any(p.get("is_known") for p in published)


def test_a_clear_majority_is_still_extracted_and_recognised(tmp_path):
    ctx, handler, engine, sent, published = _setup(tmp_path, require_match=True)
    pcm = _pcm(
        _tone(CHIME, 1.0), _tone(JONAS, 2.5), _gap(0.1), _tone(STRANGER, 0.8),
    )
    _run(handler, pcm)
    assert sent == ["Licht an"]
    assert engine.audio == [pcm]
    known = [p for p in published if p.get("is_known")]
    assert known and known[-1]["speaker"] == "Jonas"
