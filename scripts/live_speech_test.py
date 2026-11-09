# -*- coding: utf-8 -*-
"""
=============================================================================
 CHƯƠNG TRÌNH KIỂM THỬ GIỌNG NÓI & TỪ ĐỆM TRỰC TIẾP TỪ MICRO (LIVE STT TEST)
 Đề tài: Nền tảng Đánh giá Kỹ năng Thuyết trình Tự động (PBL6 - ĐHBK Đà Nẵng)
 AI & Data Lead: Nguyễn Đỗ Thắng
 Tech Lead Đồng hành: Trần Gia Bách (Bách)

 Nâng cấp v2.0:
 1. Hiển thị thang đo âm lượng Real-time (VU Meter) khi đang nói để biết mic có vào không.
 2. Tự động chuẩn hóa biên độ (Dynamic Gain Normalization) chống tiếng mic bị quá nhỏ.
 3. Tắt bộ lọc Silero VAD quá gắt làm nuốt tiếng người nói.
 4. Tự động lưu file WAV để kiểm tra lại chất lượng thu âm.
=============================================================================
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf

# Đảm bảo in UTF-8 không lỗi trên Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

FILLER_WORDS = [
    "ờ", "à", "ừm", "ừ", "kiểu như", "thì là", "như là",
    "ờm", "thì", "kiểu", "ấy", "nói chung là"
]

# Prompt mớm ngữ cảnh cho Whisper: Bắt buộc giữ từ đệm & nhận diện phương ngữ Việt Nam
DEFAULT_INITIAL_PROMPT = (
    "Sau đây là bài thuyết trình của sinh viên Trường Đại học Bách Khoa, Đại học Đà Nẵng (DUT). "
    "Bài nói mang ngữ điệu tự nhiên đa vùng miền (miền Trung, Đà Nẵng, Quảng Nam, Quảng Ngãi, Huế hoặc miền Bắc, miền Nam), "
    "chứa các thuật ngữ công nghệ thông tin và giữ nguyên vẹn các từ đệm ngập ngừng: à, ờ, ừm, ừ, kiểu như, thì là, nói chung là, dạ."
)

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "temp_live_speech")
os.makedirs(TEMP_DIR, exist_ok=True)
LAST_RECORDING_PATH = os.path.join(TEMP_DIR, "last_recording.wav")


def load_model(model_size="base", device="cpu", compute_type="int8"):
    from faster_whisper import WhisperModel
    print(f"[*] Đang nạp mô hình Faster-Whisper '{model_size}' ({compute_type}) trên {device.upper()}...")
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=4)
    print(f"[OK] Nạp mô hình thành công trong {time.perf_counter() - t0:.2f}s!\n")
    return model


def record_until_keypress(sample_rate=16000, device_index=None):
    """Thu âm giọng nói kèm VU meter thời gian thực cho đến khi nhấn Enter."""
    recorded_frames = []
    stop_event = threading.Event()
    current_rms = [0.0]

    def audio_callback(indata, frames, time_info, status):
        if not stop_event.is_set():
            recorded_frames.append(indata.copy())
            # Tính RMS để làm thước đo âm lượng thời gian thực
            rms = np.sqrt(np.mean(indata**2))
            current_rms[0] = rms

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=device_index,
        callback=audio_callback
    )

    def vu_meter_worker():
        meter_len = 24
        while not stop_event.is_set():
            rms_val = current_rms[0]
            # Chuẩn hóa giá trị RMS từ 0.0 -> 0.15 thành mức hiển thị 0 -> meter_len
            scaled = min(int((rms_val / 0.10) * meter_len), meter_len)
            bar = "=" * scaled + "-" * (meter_len - scaled)
            pct = min(int((rms_val / 0.10) * 100), 100)
            status_text = "[SIGNAL OK]" if scaled > 1 else "[NO SIGNAL]"
            sys.stdout.write(f"\r[MIC LIVE] [{bar}] {pct:3d}% | {status_text} (Nhan ENTER de dung)...")
            sys.stdout.flush()
            time.sleep(0.08)

    with stream:
        print("\n" + "=" * 75)
        print("[RECORDING] MICROPHONE DANG THU AM. VUI LONG NOI TU NHIEN...")
        print("   >> Noi binh thuong, chem tu dem va thuat ngu tuy y.")
        print("   >> Quan sat thanh do am luong de dam bao tin hieu ro.")
        print("   >> KHI NOI XONG, NHAN PHIM [ENTER] DE DUNG VA XU LY.")
        print("=" * 75)

        meter_thread = threading.Thread(target=vu_meter_worker, daemon=True)
        meter_thread.start()

        t_start = time.perf_counter()
        input()  # Chờ người dùng nhấn Enter
        stop_event.set()
        duration_sec = time.perf_counter() - t_start
        print("\n")

    if not recorded_frames:
        return None, 0.0

    audio_data = np.concatenate(recorded_frames, axis=0).flatten()
    return audio_data, duration_sec


def normalize_audio(audio_data):
    """Khuếch đại âm lượng phần mềm nếu mic thu âm quá nhỏ."""
    peak = float(np.max(np.abs(audio_data)))
    rms = float(np.sqrt(np.mean(audio_data**2)))
    
    # Nếu tín hiệu quá bé nhưng vẫn có âm thanh (> 0.0001)
    if peak > 1e-4:
        # Scale đưa peak lên 0.90 để Whisper nghe chuẩn nhất
        boost_factor = 0.90 / peak
        # Giới hạn hệ số nhân tối đa 100x tránh khuếch đại nhiễu trắng
        boost_factor = min(boost_factor, 100.0)
        normalized_data = audio_data * boost_factor
        return normalized_data, peak, rms, boost_factor
    else:
        return audio_data, peak, rms, 1.0


def analyze_transcript(text, duration_sec):
    """Phân tích văn bản, đếm từ đệm và tính tốc độ WPM."""
    words = text.strip().split()
    word_count = len(words)
    minutes = max(duration_sec / 60.0, 0.05)
    wpm = int(round(word_count / minutes))

    text_lower = text.lower()
    found_fillers = {}
    for fw in FILLER_WORDS:
        cnt = text_lower.count(fw)
        if cnt > 0:
            found_fillers[fw] = cnt

    return {
        "word_count": word_count,
        "wpm": wpm,
        "fillers": found_fillers,
        "total_fillers": sum(found_fillers.values())
    }


def live_push_to_talk(model_size="base", device_index=None, initial_prompt=DEFAULT_INITIAL_PROMPT):
    print("=" * 75)
    print("   CHƯƠNG TRÌNH TEST NÓI TRỰC TIẾP VỚI MICRO - PBL6 SPEECH ENGINE (v2.1)")
    print("=" * 75)
    if initial_prompt:
        print(f"[PROMPT CONDITIONING KICH HOAT]:\n   \"{initial_prompt}\"\n")

    model = load_model(model_size=model_size)

    session_count = 1
    while True:
        print(f"\n--- [LUOT NOI #{session_count}] ---")
        print("[ACTION] Nhan [ENTER] de BAT DAU NOI (hoac go 'q' roi Enter de thoat): ", end="")
        user_choice = input().strip()
        if user_choice.lower() == 'q':
            print("\n[+] Ket thuc phien kiem thu giong noi.")
            break

        audio_raw, duration_sec = record_until_keypress(device_index=device_index)
        if audio_raw is None or duration_sec < 0.8:
            print("[!] Doan noi qua ngan hoac chua thu duoc tin hieu. Vui long thu lai!")
            continue

        # Chuẩn hóa âm lượng chống mic nhỏ
        audio_norm, peak, rms, boost_factor = normalize_audio(audio_raw)
        
        # Lưu file WAV để kiểm tra lại khi cần
        try:
            sf.write(LAST_RECORDING_PATH, audio_norm, 16000)
            saved_msg = f" (Saved: {os.path.basename(LAST_RECORDING_PATH)})"
        except Exception:
            saved_msg = ""

        print(f"[*] Thu am: {duration_sec:.2f}s | Peak: {peak:.4f} | RMS: {rms:.4f} | Gain Boost: {boost_factor:.1f}x{saved_msg}")
        
        if peak < 0.001:
            print("[WARNING] Tin hieu am thanh tu Microphone qua nho (< 0.001)!")
            print("   -> Kiem tra xem mic co dang bi Mute hoac am luong trong Windows Sound Settings qua thap.")
            print("   -> He thong van tiep tuc phien am...")

        print("[*] Dang boc bang bang Faster-Whisper...")
        t_infer0 = time.perf_counter()

        # Bóc băng với prompt gợi ý + chống lặp từ (repetition penalty) + VAD
        segments, info = model.transcribe(
            audio_norm,
            language="vi",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            temperature=0.0,
            best_of=5,
            initial_prompt=initial_prompt,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

        full_text = []
        for seg in segments:
            full_text.append(seg.text.strip())
        transcribed_text = " ".join(full_text)
        infer_time = time.perf_counter() - t_infer0
        rtf = infer_time / duration_sec if duration_sec > 0 else 0.0

        stats = analyze_transcript(transcribed_text, duration_sec)

        # In kết quả trực quan
        print("\n" + "=" * 75)
        print("               KET QUA PHIEN AM GIONG NOI")
        print("=" * 75)
        print("[TRANSCRIPT]:")
        if transcribed_text:
            print(f'   "{transcribed_text}"')
        else:
            print('   (Khong nhan dien duoc am thanh ro rang. Vui long noi to hon hoac kiem tra mic!)')
        print("-" * 75)
        print(f"[-] Thoi luong audio    : {duration_sec:.2f}s")
        print(f"[-] Thoi gian xu ly     : {infer_time:.2f}s (RTF: {rtf:.3f} - Nhanh gap {1.0/max(rtf, 0.001):.1f}x real-time)")
        print(f"[-] Tong so tu          : {stats['word_count']} tu")
        print(f"[-] Toc do noi          : {stats['wpm']} WPM", end="")
        if 115 <= stats['wpm'] <= 145:
            print(" (Chuan nhip dieu thuyet trinh Do an)")
        elif stats['wpm'] > 155:
            print(" (Hoi nhanh - can noi cham rai hon)")
        elif stats['wpm'] > 0:
            print(" (Hoi cham - can tang nhip do)")
        else:
            print(" (Chua nhan dien duoc tu)")

        if stats['fillers']:
            filler_detail = [f"'{k}': {v} lan" for k, v in stats['fillers'].items()]
            print(f"[!] Tu dem phat hien    : {stats['total_fillers']} lan ({', '.join(filler_detail)})")
        else:
            print("[OK] Tu dem phat hien   : 0 lan (Khong bi am o)")
        print("=" * 75)

        session_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Speech-to-Text Test với Micro (v2.1 - Context Prompt)")
    parser.add_argument("--model", "-m", type=str, default="base", choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Kích cỡ model Whisper ('base' siêu tốc, 'small' chính xác cao hơn, 'medium' chất lượng tốt nhất)")
    parser.add_argument("--device", "-d", type=int, default=None,
                        help="Device ID của Microphone (chạy python -c 'import sounddevice; print(sounddevice.query_devices())')")
    parser.add_argument("--prompt", "-p", type=str, default=DEFAULT_INITIAL_PROMPT,
                        help="Prompt mớm ngữ cảnh để giữ từ đệm và nhận diện giọng vùng miền")
    args = parser.parse_args()

    live_push_to_talk(model_size=args.model, device_index=args.device, initial_prompt=args.prompt)
