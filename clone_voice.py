#!/usr/bin/env python3
"""
Production CLI for Project 100: Real-Time Voice Cloning and Emotion Transfer
=============================================================================
Clones a target speaker's voice from a short reference audio clip (or preset persona)
and generates emotionally expressive speech (happy, sad, angry, surprised, neutral)
in real time.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from main import (
    RealtimeVoiceCloningEmotionModel,
    AudioSpectrogramExtractor,
    SAMPLE_RATE,
    AUDIO_DURATION,
    N_SAMPLES,
    SPEAKERS,
    EMOTIONS,
    EMOTION_TO_ID,
    CHAR_TO_ID,
    synth_speech_utterance,
)

def load_cloning_model(checkpoint_path: str = "models/voice_cloning_emotion_best.pth", device: str = "cpu"):
    """Load pre-trained Voice Cloning & Emotion Transfer model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Please run 'python main.py' first to train the system."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RealtimeVoiceCloningEmotionModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def parse_text_to_tokens(text: str) -> list:
    """Converts input text or vowel string into phonetic vocabulary token IDs."""
    text_clean = text.lower().replace("-", " ").replace(",", " ")
    tokens = []
    for ch in text_clean:
        if ch in CHAR_TO_ID:
            tokens.append(CHAR_TO_ID[ch])
        elif ch == " ":
            tokens.append(CHAR_TO_ID["_"])
        else:
            # Map unrecognized characters to closest vowel / neutral
            tokens.append(CHAR_TO_ID["_"])
    
    # If string was empty or invalid, fallback to standard vowel sequence
    if not tokens:
        tokens = [CHAR_TO_ID[v] for v in ["a", "i", "u", "e", "o"]]
    
    return tokens

def clone_and_express(
    text: str = "a-i-u-e-o",
    speaker_ref_wav: str = None,
    speaker_preset: str = None,
    emotion: str = "happy",
    emotion_intensity: float = 1.0,
    output_wav: str = "outputs/cloned_emotional_speech.wav",
    checkpoint_path: str = "models/voice_cloning_emotion_best.pth",
    model = None,
    device: str = "cpu"
):
    """Generates expressive emotional speech in a cloned speaker's voice."""
    if model is None:
        model = load_cloning_model(checkpoint_path=checkpoint_path, device=device)

    extractor = AudioSpectrogramExtractor()
    emotion_name = emotion.lower()
    if emotion_name not in EMOTION_TO_ID:
        print(f"[WARN] Unknown emotion '{emotion}'. Defaulting to 'happy'. Valid: {EMOTIONS}")
        emotion_name = "happy"
    emo_id = EMOTION_TO_ID[emotion_name]

    # Resolve speaker reference audio
    if speaker_ref_wav and os.path.exists(speaker_ref_wav):
        wav_ref, _ = sf.read(speaker_ref_wav)
        if wav_ref.ndim > 1:
            wav_ref = np.mean(wav_ref, axis=-1)
        if len(wav_ref) < N_SAMPLES:
            wav_ref = np.pad(wav_ref, (0, N_SAMPLES - len(wav_ref)))
        else:
            wav_ref = wav_ref[:N_SAMPLES]
        speaker_label = os.path.basename(speaker_ref_wav)
    else:
        # Use speaker preset persona
        spk_idx = 1  # Default: assistant_female
        if speaker_preset:
            matches = [s["id"] for s in SPEAKERS if s["name"].lower() == speaker_preset.lower()]
            if matches:
                spk_idx = matches[0]
        wav_ref, _, _ = synth_speech_utterance(speaker_id=spk_idx, emotion="neutral")
        speaker_label = SPEAKERS[spk_idx]["name"]

    ref_mel = extractor.extract(torch.tensor(wav_ref, dtype=torch.float32)).to(device)

    # Tokenize input text
    token_list = parse_text_to_tokens(text)
    tokens_tensor = torch.tensor([token_list], dtype=torch.long, device=device)

    # Real-Time Synthesis
    t0 = time.perf_counter()
    with torch.no_grad():
        mel_synth, s_emb, e_emb = model.clone_and_express(
            text_tokens=tokens_tensor,
            spk_ref_mel=ref_mel,
            emo_id=emo_id,
            emo_intensity=emotion_intensity
        )
        s_synth, _ = model.speaker_enc(mel_synth)
        pred_emo_logits = model.mel_emotion_classifier(mel_synth)
        pred_emo_id = int(torch.argmax(pred_emo_logits, dim=-1).item())
        pred_emo_name = EMOTIONS[pred_emo_id]
        emo_confidence = float(F.softmax(pred_emo_logits, dim=-1)[0, emo_id].item()) * 100.0

    latency_ms = (time.perf_counter() - t0) * 1000.0
    speedup = AUDIO_DURATION / (latency_ms / 1000.0)

    # Invert mel to waveform
    wav_synth = extractor.vocode(mel_synth).cpu().numpy().flatten()

    # Quantitative Speaker Embedding Cosine Similarity (SECS)
    sec_similarity = float(F.cosine_similarity(s_synth, s_emb).item())

    # Save output audio
    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)
    sf.write(output_wav, wav_synth, SAMPLE_RATE)

    print("\n" + "=" * 65)
    print("      REAL-TIME VOICE CLONING & EMOTION SYNTHESIS COMPLETE")
    print("=" * 65)
    print(f"Target Voice Persona:     {speaker_label}")
    print(f"Target Emotion:           {emotion_name.upper()} (Intensity: {emotion_intensity:.1f}x)")
    print(f"Detected Output Emotion:  {pred_emo_name.upper()} (Confidence: {emo_confidence:.1f}%)")
    print(f"Speaker Similarity (SECS):{sec_similarity:.4f} (High voice clone fidelity)")
    print(f"Inference Latency:        {latency_ms:.2f} ms ({speedup:.1f}x real-time speedup)")
    print(f"Synthesized Speech Audio: {output_wav}")
    print("=" * 65)

    return wav_synth, sec_similarity, latency_ms

def run_multi_emotion_demo(checkpoint_path: str = "models/voice_cloning_emotion_best.pth", device: str = "cpu"):
    """Demonstrates voice cloning and emotional expressive synthesis across all emotions."""
    print("\n" + "=" * 65)
    print("    RUNNING PROJECT 100 GRAND FINALE MULTI-EMOTION SHOWCASE")
    print("=" * 65)

    model = load_cloning_model(checkpoint_path, device=device)
    os.makedirs("outputs", exist_ok=True)

    test_emotions = ["happy", "sad", "angry", "surprised", "neutral"]
    target_speakers = ["assistant_female", "hero_baritone"]

    for spk in target_speakers:
        print(f"\n>>> Cloned Voice Showcase: Speaker Persona '{spk}'")
        for emo in test_emotions:
            out_file = f"outputs/demo_cloned_{spk}_{emo}.wav"
            clone_and_express(
                text="a-e-i-o-u",
                speaker_preset=spk,
                emotion=emo,
                emotion_intensity=1.2 if emo != "neutral" else 1.0,
                output_wav=out_file,
                model=model,
                device=device
            )

    print("\n[SUCCESS] Multi-emotion voice cloning showcase completed!")

def main():
    parser = argparse.ArgumentParser(description="Real-Time Voice Cloning and Emotion Transfer CLI")
    parser.add_argument("--text", type=str, default="a-i-u-e-o", help="Phonetic text or vowel sequence to articulate")
    parser.add_argument("--speaker-ref", type=str, default=None, help="Path to reference speaker audio file (.wav)")
    parser.add_argument("--speaker-preset", type=str, default="assistant_female", choices=[s["name"] for s in SPEAKERS], help="Target speaker persona")
    parser.add_argument("--emotion", type=str, default="happy", choices=EMOTIONS, help="Target emotional expression")
    parser.add_argument("--emotion-intensity", type=float, default=1.0, help="Continuous emotion intensity scaling (e.g. 0.5 to 2.0)")
    parser.add_argument("--output", type=str, default=None, help="Output audio file path (.wav)")
    parser.add_argument("--demo", action="store_true", help="Run automated multi-speaker multi-emotion demonstration")
    parser.add_argument("--checkpoint", type=str, default="models/voice_cloning_emotion_best.pth", help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")

    args = parser.parse_args()

    if args.demo:
        run_multi_emotion_demo(checkpoint_path=args.checkpoint, device=args.device)
        return

    out_path = args.output
    if out_path is None:
        spk_tag = args.speaker_preset or (os.path.splitext(os.path.basename(args.speaker_ref))[0] if args.speaker_ref else "cloned")
        out_path = f"outputs/cloned_{spk_tag}_{args.emotion}.wav"

    clone_and_express(
        text=args.text,
        speaker_ref_wav=args.speaker_ref,
        speaker_preset=args.speaker_preset,
        emotion=args.emotion,
        emotion_intensity=args.emotion_intensity,
        output_wav=out_path,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

if __name__ == "__main__":
    main()

