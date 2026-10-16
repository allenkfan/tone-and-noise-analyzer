"""The LLM judge with the Gemini client mocked. Never touches the network or needs a key."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import errors, types

from analyzer import llm
from analyzer.schema import TONES

SR = 16_000
ANSWER = {"rationale": "The caller asks about a refund calmly.", "emotional_tone": "frustrated",
          "emotional_intensity": "medium", "background_noise_type": "Television ",
          "speaker_overlap_present": True, "confidence": 0.7}


def usage(audio=960, text=600, output=80, thoughts=0):
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=audio + text, candidates_token_count=output, thoughts_token_count=thoughts,
        prompt_tokens_details=[types.ModalityTokenCount(modality="AUDIO", token_count=audio),
                               types.ModalityTokenCount(modality="TEXT", token_count=text)])


class FakeClient:
    """Stands in for genai.Client: replies come from a list; an exception in the list is raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.models = self

    def generate_content(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(text=text, usage_metadata=usage())


@pytest.fixture
def clip():
    t = np.arange(2 * SR) / SR
    return np.repeat((0.1 * np.sin(2 * np.pi * 220 * t)).astype("f4")[None], 2, axis=0)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_breaker", llm._Breaker())
    monkeypatch.setattr(llm, "CALIBRATION", {})

    def install(*replies):
        client = FakeClient(*replies)
        monkeypatch.setattr(llm, "_client", lambda key: client)
        return client
    return install


def test_no_key_raises_without_calling(monkeypatch, clip):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(llm, "_client", lambda key: pytest.fail("client created without a key"))
    with pytest.raises(llm.LLMUnavailable):
        llm.judge(clip, SR)


def test_falls_back_to_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "generic")
    assert llm.api_key() == "generic"


def test_judge_parses_answer_and_costs_it(fake, clip):
    client = fake(ANSWER)
    j = llm.judge(clip, SR)
    assert (j.emotional_tone, j.emotional_intensity, j.speaker_overlap_present) == ("frustrated", "medium", True)
    assert j.background_noise_type == "television"
    assert j.confidence == 0.7
    spec = llm.MODELS[llm.MODEL]
    expected = (600 * spec.text_usd + 960 * spec.audio_usd + 80 * spec.output_usd) / 1e6
    assert j.usage["usd"] == pytest.approx(expected)
    assert j.usage["audio_tokens"] == 960 and j.usage["model"] == llm.MODEL

    request = client.requests[0]
    assert request["model"] == llm.MODEL
    audio = request["contents"][0].parts[0]
    assert audio.inline_data.mime_type == "audio/ogg"
    assert audio.inline_data.data[:4] == b"OggS"
    assert request["config"].response_json_schema["properties"]["emotional_tone"]["enum"] == list(TONES)


def test_acoustics_go_in_the_prompt_and_no_noise_empties_the_type(fake, clip):
    client = fake(ANSWER)
    acoustics = {"snr_db": None, "noise_tags": [], "background_noise_present": False, "new_key": 1}
    j = llm.judge(clip, SR, acoustics)
    texts = " ".join(p.text for p in client.requests[0]["contents"][0].parts if p.text)
    assert '"new_key": 1' in texts and "empty background_noise_type" in texts
    assert j.background_noise_type == ""


def test_schema_failure_is_retried_once(fake, clip):
    client = fake("not json", {**ANSWER, "emotional_tone": "angry"}, ANSWER)
    with pytest.raises(llm.LLMUnavailable):
        llm.judge(clip, SR)
    assert len(client.requests) == 2

    client = fake("{}", ANSWER)
    assert llm.judge(clip, SR).emotional_tone == "frustrated"


def test_api_error_becomes_unavailable(fake, clip):
    fake(errors.ServerError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}}))
    with pytest.raises(llm.LLMUnavailable, match="503"):
        llm.judge(clip, SR)


def test_breaker_skips_the_api_after_repeated_failures(fake, clip):
    failure = errors.ClientError(403, {"error": {"message": "bad key", "status": "PERMISSION_DENIED"}})
    client = fake(*[failure] * llm.BREAKER_FAILURES)
    for _ in range(llm.BREAKER_FAILURES):
        with pytest.raises(llm.LLMUnavailable):
            llm.judge(clip, SR)
    with pytest.raises(llm.LLMUnavailable, match="skipped"):
        llm.judge(clip, SR)
    assert len(client.requests) == llm.BREAKER_FAILURES


def test_mono_input_is_accepted(clip):
    assert llm.encode_audio(clip[0], SR)[:4] == b"OggS"


def test_calibration_interpolates(monkeypatch):
    monkeypatch.setattr(llm, "CALIBRATION", {llm.MODEL: [(0.0, 0.2), (1.0, 0.6)]})
    assert llm.calibrate(0.5) == pytest.approx(0.4)
    assert llm.calibrate(0.5, "gemini-3.8-flash") == 0.5


def test_default_model_is_calibrated(fake, clip, monkeypatch):
    monkeypatch.setattr(llm, "CALIBRATION", {llm.MODEL: [(0.5, 0.1), (1.0, 0.3)]})
    fake(ANSWER)
    j = llm.judge(clip, SR)
    assert j.confidence == pytest.approx(0.18) and j.usage["raw_confidence"] == 0.7


def test_dev_runner_scores_a_labeled_folder(fake, batch_dir, monkeypatch, tmp_path, capsys):
    fake(*[ANSWER] * 4)
    monkeypatch.chdir(tmp_path)
    llm.main([str(batch_dir), "--workers", "1", "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "3 judged, 1 failed" in out  # corrupt.wav fails to decode
    assert "Projected" in out and "emotional_tone" in out
    lines = (tmp_path / "out" / f"{llm.MODEL}.jsonl").read_text().splitlines()
    assert len(lines) == 4


def test_network_error_becomes_unavailable(fake, clip):
    import httpx

    fake(httpx.ReadTimeout("timed out"))
    with pytest.raises(llm.LLMUnavailable, match="ReadTimeout"):
        llm.judge(clip, SR)


def test_isotonic_fit_is_monotone_and_pools_violators():
    raw = [0.5, 0.5, 0.7, 0.7, 0.9, 0.9, 0.95]
    correct = [False, True, True, True, False, True, True]
    points = llm.fit_calibration(raw, correct)
    assert [x for x, _ in points] == [0.5, 0.7, 0.9, 0.95]
    ys = [y for _, y in points]
    assert ys == sorted(ys) and ys[0] == 0.5 and ys[1] == ys[2] == 0.75


def test_calibration_error():
    assert llm.calibration_error([0.9] * 10, [True] * 9 + [False]) == pytest.approx(0.0)
    assert llm.calibration_error([0.9] * 4, [False] * 4) == pytest.approx(0.9)
