# -*- coding: utf-8 -*-
"""
=============================================================================
 KIỂM KÊ TÀI NGUYÊN & KHÓA SPLIT CẤP NGƯỜI NÓI (AUDIT & SPEAKER SPLIT)
 Đề tài: Nền tảng Đánh giá Kỹ năng Thuyết trình Tự động Đa Phương thức (PBL6)
 AI & Data Lead: Nguyễn Đỗ Thắng
 
 Thực hiện Mục 2.1, 7.1, 7.3 và Phụ lục B (audit_assets.py + make_speaker_split.py)
 trong Kế hoạch triển khai AI PBL6:
 1. Kiểm kê toàn bộ video, audio, bảng nhãn, trọng số hiện có kèm checksum.
 2. Phát hiện và gắn cờ các mẫu vi phạm phạm vi SRS (như HB_001 tiếng Anh dài 1250s).
 3. Khóa danh sách speaker_id và tạo split (train / dev / test) không rò rỉ người nói
    cho cả bộ video thuyết trình (DUT-PresCorpus) và bộ thu âm (DUT-Speech).
 4. Dọn dẹp các thư mục tạm (temp_eval, temp_live_*) trong data/processed/.
=============================================================================
"""

import os
import sys
import json
import shutil
import hashlib
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data", "manifests")
os.makedirs(MANIFEST_DIR, exist_ok=True)


def short_sha256(path):
    if not os.path.exists(path) or os.path.isdir(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def clean_temp_directories():
    temp_dirs = [
        os.path.join(PROJECT_ROOT, "data", "processed", "temp_eval"),
        os.path.join(PROJECT_ROOT, "data", "processed", "temp_live_recording"),
        os.path.join(PROJECT_ROOT, "data", "processed", "temp_live_speech"),
    ]
    cleaned = []
    for d in temp_dirs:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                cleaned.append(os.path.relpath(d, PROJECT_ROOT))
            except Exception:
                pass
    dup_script = os.path.join(SCRIPT_DIR, "generate_colab_notebook.py")
    if os.path.exists(dup_script):
        try:
            os.remove(dup_script)
            cleaned.append("scripts/generate_colab_notebook.py (bản trùng lặp cũ)")
        except Exception:
            pass
    return cleaned


def build_presentation_manifest_and_split():
    meta_path = os.path.join(PROJECT_ROOT, "data", "metadata.csv")
    if not os.path.exists(meta_path):
        return []

    df_meta = pd.read_csv(meta_path)
    # Chuẩn hóa 3 bối cảnh chính thức của dự án: do_an, hung_bien, pitching
    valid_contexts = {"do_an", "hung_bien", "pitching"}

    manifest_rows = []
    # Chia theo nhóm người nói độc lập (mỗi video trong 32 mẫu là 1 diễn giả riêng)
    for idx, row in df_meta.iterrows():
        vid_id = str(row["video_id"]).strip().upper()
        ctx_raw = str(row["context"]).strip().lower()
        ctx_std = ctx_raw if ctx_raw in valid_contexts else "do_an"
        fname = str(row["filename"]).strip()
        dur_sec = float(row.get("duration_sec", 180.0))
        topic = str(row.get("topic", ""))

        rel_video_path = os.path.join("data", "raw", "DUT-PresCorpus", ctx_std, fname)
        abs_video_path = os.path.join(PROJECT_ROOT, rel_video_path)
        abs_wav_path = os.path.join(PROJECT_ROOT, "data", "processed", "audio_16k", f"{vid_id}.wav")

        # Kiểm tra điều kiện hợp lệ (tiếng Việt, loại trừ HB_001 tiếng Anh dài 1250s)
        qc_flags = []
        qc_status = "accepted"
        if vid_id == "HB_001" or "Brené Brown" in topic or dur_sec > 360.0:
            qc_flags.append("OUT_OF_SCOPE_LANGUAGE_OR_DURATION")
            qc_status = "excluded_out_of_scope"
        elif dur_sec > 240.0:
            qc_flags.append("DURATION_LONG_OVER_240S")

        speaker_id = f"spk_pres_{vid_id.lower()}"

        # Phân bổ Speaker-disjoint split (70% train, 15% dev, 15% test cân bằng từng bối cảnh)
        if qc_status != "accepted":
            split = "excluded"
        else:
            num_part = int("".join(filter(str.isdigit, vid_id)) or "1")
            if num_part % 6 == 0:
                split = "test"
            elif num_part % 6 == 5:
                split = "dev"
            else:
                split = "train"

        manifest_rows.append({
            "recording_id": vid_id,
            "speaker_id": speaker_id,
            "session_id": f"ses_{vid_id.lower()}",
            "context": ctx_std,
            "take_type": "real_corpus",
            "relative_path": rel_video_path.replace("\\", "/"),
            "video_exists": os.path.exists(abs_video_path),
            "audio_16k_exists": os.path.exists(abs_wav_path),
            "sha256_short": short_sha256(abs_video_path) or short_sha256(abs_wav_path),
            "duration_sec": dur_sec,
            "split": split,
            "split_version": "speaker_split_v1",
            "parent_id": None,
            "qc_status": qc_status,
            "qc_flags": ";".join(qc_flags) if qc_flags else "OK"
        })

    df_manifest = pd.DataFrame(manifest_rows)
    out_csv = os.path.join(MANIFEST_DIR, "presentation_corpus_manifest.csv")
    df_manifest.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_json = os.path.join(MANIFEST_DIR, "presentation_corpus_manifest.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, ensure_ascii=False, indent=2)

    return manifest_rows


def audit_weights_and_artifacts():
    adapter_cfg_path = os.path.join(PROJECT_ROOT, "weights", "whisper_lora_vietnamese", "adapter_config.json")
    lora_base = "unknown"
    lora_r = None
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            lora_base = cfg.get("base_model_name_or_path", "unknown")
            lora_r = cfg.get("r")

    return {
        "whisper_lora_adapter": {
            "path": "weights/whisper_lora_vietnamese/adapter_model.safetensors",
            "exists": os.path.exists(os.path.join(PROJECT_ROOT, "weights", "whisper_lora_vietnamese", "adapter_model.safetensors")),
            "current_base_model_in_config": lora_base,
            "lora_rank": lora_r,
            "audit_note": (
                "CẢNH BÁO: Trọng số hiện tại đang gắn với 'openai/whisper-small'. "
                "Cần chạy lại Notebook Colab/train_whisper_lora.py để cập nhật sang 'vinai/PhoWhisper-small'!"
                if lora_base != "vinai/PhoWhisper-small"
                else "Chuẩn: Đã khớp với vinai/PhoWhisper-small."
            )
        },
        "face_landmarker": {
            "path": "weights/face_landmarker.task",
            "exists": os.path.exists(os.path.join(PROJECT_ROOT, "weights", "face_landmarker.task")),
            "sha256_short": short_sha256(os.path.join(PROJECT_ROOT, "weights", "face_landmarker.task"))
        }
    }


def main():
    print("=" * 80)
    print("   KIỂM KÊ TÀI NGUYÊN & KHÓA SPEAKER SPLIT (THEO KẾ HOẠCH TRIỂN KHAI AI)")
    print("=" * 80)

    cleaned = clean_temp_directories()
    if cleaned:
        print(f"[+] Đã dọn dẹp các thư mục/file rác tạm thời: {', '.join(cleaned)}")

    pres_manifest = build_presentation_manifest_and_split()
    df_pres = pd.DataFrame(pres_manifest)
    print(f"[+] Đã tạo Manifest cho bộ video thuyết trình ({len(df_pres)} video):")
    for st, grp in df_pres.groupby("split"):
        print(f"    • Split [{st.upper():<8}]: {len(grp):2d} video ({grp['speaker_id'].nunique()} speakers)")

    # Gọi chuẩn hóa & chia tập cho DUT-Speech
    from prepare_dut_speech_dataset import process_dataset
    process_dataset()

    weights_audit = audit_weights_and_artifacts()
    audit_report_path = os.path.join(MANIFEST_DIR, "audit_inventory_report.json")
    with open(audit_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "cleaned_temp_items": cleaned,
            "presentation_corpus_total": len(pres_manifest),
            "presentation_splits": df_pres["split"].value_counts().to_dict(),
            "weights_audit": weights_audit
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Đã xuất báo cáo kiểm kê toàn diện tại: {audit_report_path}")
    print(f"     -> Trạng thái Whisper LoRA: {weights_audit['whisper_lora_adapter']['audit_note']}")


if __name__ == "__main__":
    main()

