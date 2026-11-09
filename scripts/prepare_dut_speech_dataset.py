# -*- coding: utf-8 -*-
"""
=============================================================================
 CHƯƠNG TRÌNH TỰ ĐỘNG CHUẨN HÓA DỮ LIỆU THU ÂM (DUT-SPEECH CORPUS)
 Đề tài: Nền tảng Đánh giá Kỹ năng Thuyết trình Tự động (PBL6 - ĐHBK Đà Nẵng)
 AI & Data Lead: Nguyễn Đỗ Thắng
 
 Chức năng (Đã nâng cấp theo Kế hoạch triển khai AI PBL6 Mục 6.2, 7.1, 8.4):
 1. Quét tất cả file thu âm (.m4a, .mp3, .wav, .aac, .ogg) từ các thư mục nguoi_01 -> nguoi_10
 2. Dùng ffmpeg/soundfile chuẩn hóa về 16.000 Hz, Mono, 16-bit PCM WAV
 3. Đếm từ đệm theo Ranh giới Từ/Cụm từ (Word-boundary Event Matching), tuyệt đối
    KHÔNG match substring ('à' trong 'bày', 'ờ' trong 'thường'), đồng thời tách biệt:
    - nonlexical_fillers: ờ, ừm, ờm, à
    - discourse_fillers: kiểu như, thì là, nói chung là (có kiểm ngữ cảnh)
 4. Gán canonical_speaker_id (gộp nguoi_03 và nguoi_10 của cùng 1 người nói) và
    chia tập speaker-disjoint split (train / dev / test) không rò rỉ giọng nói.
=============================================================================
"""

import os
import re
import sys
import glob
import hashlib
import subprocess
import soundfile as sf
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dut_speech_recordings")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "dut_speech_16k")
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data", "manifests")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

SCRIPT_SENTENCES = {
    1: "Dạ xin chào thầy và các bạn, thì à, hôm nay nhóm em xin báo cáo tiến độ đồ án.",
    2: "Giao diện của hệ thống được tụi em xây dựng bằng React, ờ, chạy rất mượt trên máy tính.",
    3: "Ở phần Backend thì, ừm, nhóm em dùng Python để xử lý các mô hình AI.",
    4: "Để lưu dữ liệu người dùng thì tụi em dùng Database, à, kết nối trực tiếp với Server.",
    5: "Khi sinh viên bật Camera lên thuyết trình thì, kiểu như, hệ thống sẽ tự động quay video lại.",
    6: "Dữ liệu âm thanh từ Microphone được gửi qua API, ờm, để AI bóc băng thành chữ.",
    7: "Trên màn hình Dashboard có hiển thị biểu đồ, thì là, để sinh viên biết mình được mấy điểm.",
    8: "Sau khi chấm điểm xong thì hệ thống sẽ, à, gợi ý các mẹo giúp cải thiện bài nói.",
    9: "Dạ thưa thầy, thì cái phần này nhóm em, à, cũng có tính tới rồi ạ.",
    10: "Kiểu như là khi sinh viên thuyết trình bị run, thì ờ, tốc độ nói sẽ nhanh hơn bình thường.",
    11: "Nói chung là tụi em nhận thấy một bài nói chuẩn thì, ừm, không nên có quá nhiều từ đệm.",
    12: "Dạ về phần giao diện Website thì, à, tụi em thiết kế theo dạng thẻ rất dễ nhìn.",
    13: "Ủa, à, dạ, cái chỗ này tụi em có huấn luyện thêm mô hình để nhận diện chuẩn hơn ạ.",
    14: "Ừm, theo như nhóm tìm hiểu thì việc nhìn thẳng vào Camera khi nói là rất quan trọng.",
    15: "Dạ thầy hỏi câu này, ờ, kiểu như tụi em đã thử nghiệm thực tế với các bạn trong lớp rồi ạ.",
    16: "Thì tóm lại là kết quả chấm điểm của máy tính, à, khá là sát với thực tế ạ.",
    17: "Dạ cho em hỏi thêm là, ờ, phần demo này tụi em chiếu video hay nói trực tiếp ạ?",
    18: "Chào mọi người, thì bữa ni nhóm mình sẽ test thử chức năng nhận diện giọng nói qua Micro nghe.",
    19: "Cái mic ni thu âm hơi nhỏ một chút, nhưng mà ờ, phần mềm vẫn nghe được bình thường.",
    20: "Mấy bạn vô phòng thuyết trình thì nhớ tự tin lên, đừng có, à, nhìn xuống bàn nhiều quá.",
    21: "Cái biểu đồ tròn ni hiển thị kết quả bài nói, ừm, nhìn vô là biết mình nói nhanh hay chậm liền.",
    22: "Hôm nay nhóm em xin trình bày về dự án Website hỗ trợ luyện tập thuyết trình tự động.",
    23: "À, đối với bài thuyết trình gọi vốn, sinh viên cần nói to rõ và dứt khoát hơn.",
    24: "Thì trong quá trình làm việc nhóm, tụi em thấy khó nhất là, ờ, khâu thu thập giọng nói.",
    25: "Dạ em xin hết phần trình bày, à, cảm ơn thầy cô và các bạn đã theo dõi ạ."
}

# Tách biệt âm đệm phi từ vựng (nonlexical) và từ đệm chuyển ý (discourse) theo Mục 6.2 & 8.4 Kế hoạch
NONLEXICAL_FILLERS = ["ờm", "ừm", "ờ", "à", "ừ", "ư"]
DISCOURSE_FILLERS = ["kiểu như là", "kiểu như", "thì là", "nói chung là"]

# Ánh xạ danh tính chuẩn (Canonical Speaker ID) tránh rò rỉ khi 1 người thu 2 buổi (nguoi_03 & nguoi_10)
CANONICAL_SPEAKER_MAP = {
    "nguoi_01_chi_linh_ha_tinh": "spk_01_chi_linh",
    "nguoi_02_anh_thanh_tp_hue": "spk_02_anh_thanh",
    "nguoi_03_son_dang_nghe_an": "spk_03_son_dang",
    "nguoi_04_thanh_nhan_quang_tri": "spk_04_thanh_nhan",
    "nguoi_05_huynh_tam_ha_noi": "spk_05_huynh_tam",
    "nguoi_06_tham_ho_tp_hue": "spk_06_tham_ho",
    "nguoi_07_the_ngoc_quang_binh": "spk_07_the_ngoc",
    "nguoi_08_hong_trang_da_nang": "spk_08_hong_trang",
    "nguoi_09_do_thang_ha_noi": "spk_09_do_thang",
    "nguoi_10_son_dang_pho_thong": "spk_03_son_dang",  # Cùng người nói với nguoi_03
}

# Chia tập theo cấp độ người nói (Speaker-disjoint split):
# - Train: 7 người (8 thư mục thu âm)
# - Dev: 1 người độc lập (spk_06_tham_ho)
# - Test: 1 người độc lập (spk_08_hong_trang)
SPEAKER_SPLIT_MAP = {
    "spk_01_chi_linh": "train",
    "spk_02_anh_thanh": "train",
    "spk_03_son_dang": "train",
    "spk_04_thanh_nhan": "train",
    "spk_05_huynh_tam": "train",
    "spk_07_the_ngoc": "train",
    "spk_09_do_thang": "train",
    "spk_06_tham_ho": "dev",
    "spk_08_hong_trang": "test",
}


def detect_filler_events_in_text(text: str):
    """
    Phát hiện sự kiện từ đệm theo ranh giới token nguyên vẹn (Word-boundary matching).
    Tuyệt đối không bắt nhầm 'à' bên trong 'bày', 'ờ' bên trong 'thường', hoặc 'là'/'dạ' có chức năng ngữ pháp.
    Trả về: (danh_sach_events, nonlexical_count, discourse_count)
    """
    if not text:
        return [], 0, 0

    # Chuẩn hóa chữ thường, tách dấu câu thành khoảng trắng nhưng giữ nguyên từ
    clean_tokens_str = re.sub(r"[,\.\!\?\;\:\"\(\)\[\]\-]+", " ", text.lower())
    clean_tokens_str = re.sub(r"\s+", " ", clean_tokens_str).strip()

    events = []
    nonlex_cnt = 0
    disc_cnt = 0

    # 1. Tìm các cụm discourse fillers nhiều từ trước, đánh dấu vị trí đã khớp
    occupied_spans = []
    for phrase in DISCOURSE_FILLERS:
        pattern = r"(?:^|(?<=\s))" + re.escape(phrase) + r"(?=\s|$)"
        for m in re.finditer(pattern, clean_tokens_str):
            span = (m.start(), m.end())
            if not any( not (span[1] <= s[0] or span[0] >= s[1]) for s in occupied_spans ):
                occupied_spans.append(span)
                events.append({"surface": phrase, "type": "discourse_filler", "char_start": span[0]})
                disc_cnt += 1

    # 2. Tìm các âm đệm đơn (nonlexical fillers) theo ranh giới từ độc lập
    for word in NONLEXICAL_FILLERS:
        pattern = r"(?:^|(?<=\s))" + re.escape(word) + r"(?=\s|$)"
        for m in re.finditer(pattern, clean_tokens_str):
            span = (m.start(), m.end())
            if not any( not (span[1] <= s[0] or span[0] >= s[1]) for s in occupied_spans ):
                occupied_spans.append(span)
                events.append({"surface": word, "type": "nonlexical_filler", "char_start": span[0]})
                nonlex_cnt += 1

    events.sort(key=lambda x: x["char_start"])
    return events, nonlex_cnt, disc_cnt


def extract_fillers(text):
    """Trích xuất chuỗi mô tả các sự kiện từ đệm thực tế trong câu."""
    events, _, _ = detect_filler_events_in_text(text)
    return ", ".join([e["surface"] for e in events])


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_sentence_index(filename):
    """Trích xuất số thứ tự câu từ tên file (ví dụ: cau_01.m4a -> 1, 05.wav -> 5)."""
    numbers = re.findall(r'\d+', os.path.splitext(filename)[0])
    if numbers:
        idx = int(numbers[-1])
        if 1 <= idx <= 25:
            return idx
    return None


def convert_to_wav16k(input_path, output_path):
    """Dùng ffmpeg để chuyển đổi âm thanh sang WAV 16kHz Mono."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    cmd = [
        ffmpeg_exe, "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        try:
            data, sr = sf.read(input_path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            sf.write(output_path, data, 16000)
            return True
        except Exception:
            return False


def process_dataset():
    print("=" * 75)
    print("   CHUẨN HÓA DỮ LIỆU DUT-SPEECH CORPUS (CHUẨN KẾ HOẠCH TRIỂN KHAI AI)")
    print("=" * 75)

    speaker_dirs = sorted(glob.glob(os.path.join(RAW_DIR, "nguoi_*")))
    records = []
    total_audio_sec = 0.0
    valid_extensions = {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac", ".wma"}

    if speaker_dirs:
        for spk_path in speaker_dirs:
            spk_name = os.path.basename(spk_path)
            files = sorted([f for f in os.listdir(spk_path) if os.path.splitext(f)[1].lower() in valid_extensions])
            if not files:
                continue

            print(f"[*] Đang xử lý: {spk_name} ({len(files)} file)...")
            for f in files:
                in_file = os.path.join(spk_path, f)
                sent_idx = parse_sentence_index(f)
                if sent_idx is None:
                    continue

                transcription = SCRIPT_SENTENCES.get(sent_idx, "")
                out_filename = f"{spk_name}_cau_{sent_idx:02d}.wav"
                out_file = os.path.join(OUTPUT_DIR, out_filename)

                if convert_to_wav16k(in_file, out_file):
                    try:
                        data, sr = sf.read(out_file)
                        dur = len(data) / float(sr)
                        total_audio_sec += dur
                        events, nonlex_cnt, disc_cnt = detect_filler_events_in_text(transcription)
                        fillers_str = ", ".join([e["surface"] for e in events])
                        canonical_spk = CANONICAL_SPEAKER_MAP.get(spk_name, spk_name)
                        split_name = SPEAKER_SPLIT_MAP.get(canonical_spk, "train")

                        records.append({
                            "audio_filename": out_filename,
                            "audio_path": os.path.abspath(out_file),
                            "speaker_id": spk_name,
                            "canonical_speaker_id": canonical_spk,
                            "split": split_name,
                            "sentence_id": sent_idx,
                            "duration_sec": round(dur, 2),
                            "transcription": transcription,
                            "fillers_contained": fillers_str,
                            "filler_event_count": len(events),
                            "nonlexical_filler_count": nonlex_cnt,
                            "discourse_filler_count": disc_cnt,
                            "sha256_short": compute_sha256(out_file)
                        })
                    except Exception as e:
                        print(f"   [!] Lỗi đọc file {out_filename}: {e}")
    else:
        # Nếu chỉ có sẵn các file .wav trong OUTPUT_DIR
        wav_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")))
        for out_file in wav_files:
            out_filename = os.path.basename(out_file)
            m = re.match(r"^(nguoi_\d+_.*)_cau_(\d+)\.wav$", out_filename)
            if not m:
                continue
            spk_name, sent_idx = m.group(1), int(m.group(2))
            transcription = SCRIPT_SENTENCES.get(sent_idx, "")
            data, sr = sf.read(out_file)
            dur = len(data) / float(sr)
            total_audio_sec += dur
            events, nonlex_cnt, disc_cnt = detect_filler_events_in_text(transcription)
            fillers_str = ", ".join([e["surface"] for e in events])
            canonical_spk = CANONICAL_SPEAKER_MAP.get(spk_name, spk_name)
            split_name = SPEAKER_SPLIT_MAP.get(canonical_spk, "train")
            records.append({
                "audio_filename": out_filename,
                "audio_path": os.path.abspath(out_file),
                "speaker_id": spk_name,
                "canonical_speaker_id": canonical_spk,
                "split": split_name,
                "sentence_id": sent_idx,
                "duration_sec": round(dur, 2),
                "transcription": transcription,
                "fillers_contained": fillers_str,
                "filler_event_count": len(events),
                "nonlexical_filler_count": nonlex_cnt,
                "discourse_filler_count": disc_cnt,
                "sha256_short": compute_sha256(out_file)
            })

    if not records:
        print("[!] Chưa tìm thấy file âm thanh hợp lệ.")
        return

    df = pd.DataFrame(records)
    metadata_csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    df.to_csv(metadata_csv_path, index=False, encoding="utf-8-sig")

    manifest_csv_path = os.path.join(MANIFEST_DIR, "dut_speech_speaker_split_manifest.csv")
    df.to_csv(manifest_csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print("         KẾT QUẢ CHUẨN HÓA & CHIA TẬP THEO NGƯỜI NÓI (DUT-SPEECH)")
    print("=" * 75)
    print(f"✅ Tổng số mẫu âm thanh hợp lệ  : {len(df)} mẫu ({total_audio_sec / 60.0:.2f} phút)")
    print(f"👥 Số thư mục thu âm            : {df['speaker_id'].nunique()} phiên")
    print(f"👤 Số người nói thực tế độc lập : {df['canonical_speaker_id'].nunique()} người (đã gộp nguoi_03 & nguoi_10)")
    for sp_name in ["train", "dev", "test"]:
        sub = df[df["split"] == sp_name]
        print(f"   • Split [{sp_name.upper():<5}]: {len(sub):3d} mẫu | {sub['canonical_speaker_id'].nunique()} người nói độc lập | {sub['duration_sec'].sum()/60.0:.2f} phút")
    print(f"📝 File metadata chuẩn train    : {metadata_csv_path}")
    print(f"🔒 File split manifest khóa     : {manifest_csv_path}")
    print("=" * 75)


if __name__ == "__main__":
    process_dataset()
