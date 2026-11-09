# -*- coding: utf-8 -*-
"""
=============================================================================
 KIỂM THỬ MÔ HÌNH WHISPER LORA TIẾNG VIỆT ĐÃ HUẤN LUYỆN
 Đề tài: Nền tảng Đánh giá Kỹ năng Thuyết trình Tự động (PBL6 - ĐHBK Đà Nẵng)
 AI & Data Lead: Nguyễn Đỗ Thắng

 Đã nâng cấp theo Kế hoạch triển khai AI PBL6 (Mục 8.4 & 9):
 - Đếm sự kiện từ đệm theo ranh giới từ (Word-boundary Event Matching), tách biệt
   nonlexical_filler ('ờ', 'ừm', 'à', 'ờm') và discourse_filler ('kiểu như', 'thì là'),
   tuyệt đối không đếm nhầm substring ('à' trong 'bày', 'ờ' trong 'thường').
 - Mặc định kiểm thử trên mẫu của người nói thuộc tập Test độc lập (nguoi_08_hong_trang_da_nang).
=============================================================================
"""

import os
import sys
import time
import json

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from prepare_dut_speech_dataset import detect_filler_events_in_text

LORA_PATH = os.path.join(PROJECT_ROOT, "weights", "whisper_lora_vietnamese")


def test_inference(audio_path=None):
    print("=" * 75)
    print("      KIỂM THỬ SUY LUẬN MÔ HÌNH WHISPER LORA TIẾNG VIỆT (PBL6)")
    print("=" * 75)

    if not os.path.exists(LORA_PATH):
        print(f"[!] Lỗi: Không tìm thấy thư mục trọng số LoRA tại: {LORA_PATH}")
        return

    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from peft import PeftModel
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        print(f"\n[!] Thiếu thư viện deep learning: {e}")
        print("    >> Cài đặt: pip install peft transformers soundfile")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị tính toán (Device): {device.upper()}")

    BASE_MODEL = "vinai/PhoWhisper-small"
    cfg_path = os.path.join(LORA_PATH, "adapter_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                BASE_MODEL = cfg.get("base_model_name_or_path", BASE_MODEL)
        except Exception:
            pass

    print(f"[*] 1. Đang nạp mô hình nền tảng: {BASE_MODEL}...")
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    print(f"[*] 2. Đang gắn LoRA Adapter từ: {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    processor = WhisperProcessor.from_pretrained(LORA_PATH)
    model.to(device)
    model.eval()
    print("[+] Nạp thành công mô hình LoRA!")

    if audio_path is None or not os.path.exists(audio_path):
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "processed", "dut_speech_16k", "nguoi_08_hong_trang_da_nang_cau_15.wav"),
            os.path.join(PROJECT_ROOT, "data", "processed", "dut_speech_16k", "nguoi_08_hong_trang_da_nang_cau_01.wav"),
            os.path.join(PROJECT_ROOT, "data", "processed", "dut_speech_16k", "nguoi_01_chi_linh_ha_tinh_cau_01.wav")
        ]
        for c in candidates:
            if os.path.exists(c):
                audio_path = c
                break

    if not audio_path or not os.path.exists(audio_path):
        print("[!] Chưa tìm thấy file audio mẫu .wav để test.")
        return

    print(f"\n[*] 3. Đang xử lý file âm thanh (Tập Test độc lập): {os.path.basename(audio_path)}")
    audio_data, sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    duration_sec = len(audio_data) / float(sr)
    print(f"    - Thời lượng file: {duration_sec:.2f} giây")

    start_time = time.perf_counter()
    input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="vi", task="transcribe")

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    latency = time.perf_counter() - start_time

    events, nonlex_cnt, disc_cnt = detect_filler_events_in_text(transcription)
    filler_summary = [f"'{e['surface']}' ({e['type']})" for e in events]

    print("\n" + "=" * 75)
    print("      KẾT QUẢ PHIÊN ÂM & PHÁT HIỆN TỪ ĐỆM THEO SỰ KIỆN (EVENT)")
    print("=" * 75)
    print(f"[TRANSCRIPT]        : \"{transcription}\"")
    print(f"[LATENCY]           : {latency:.2f} giây (RTF: {latency/max(duration_sec, 0.1):.3f})")
    print(f"[NONLEXICAL FILLERS]: {nonlex_cnt} sự kiện (ờ, ừm, à, ờm)")
    print(f"[DISCOURSE FILLERS] : {disc_cnt} sự kiện (kiểu như, thì là, nói chung là)")
    print(f"[CHI TIẾT EVENTS]   : {', '.join(filler_summary) if filler_summary else 'Không phát hiện sự kiện từ đệm'}")
    print("=" * 75)


if __name__ == "__main__":
    test_inference()
