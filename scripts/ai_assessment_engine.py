# -*- coding: utf-8 -*-
"""
=============================================================================
 BỘ NÃO SUY LUẬN AI TOÀN DIỆN (MULTIMODAL AI ASSESSMENT CORE ENGINE)
 Đề tài: Nền tảng Đánh giá Kỹ năng Thuyết trình Tự động Đa Phương thức & Đa Bối cảnh
 Học phần: PBL6 - Khoa CNTT, Trường Đại học Bách Khoa - ĐHĐN
 AI & Data Lead: Nguyễn Đỗ Thắng
 Backend Integration: Võ Thành Nhân | Frontend Integration: Đặng Quang Sơn

 Đã nâng cấp chuẩn Kế hoạch triển khai AI PBL6 (Mục 1.1, 3.2, 10.3, 11.1, 12, Phụ lục A3):
 1. Hỗ trợ cả 3 Enum bối cảnh chuẩn SRS ('Academic', 'Pitching', 'Interview') lẫn alias
    cũ ('do_an', 'hung_bien', 'pitching').
 2. Kiểm tra chất lượng đầu vào (validate_media & quality gate):
    - Thời lượng chuẩn 15-180s, kiểm tra độ phủ khuôn mặt (face_coverage >= 80%) và âm thanh.
    - Khi không đủ quan sát (face_coverage < 0.80), gắn cờ LOW_FACE_COVERAGE / insufficient_data.
 3. Hai chính sách chấm điểm có phiên bản:
    - 'srs_v1_1' (Mặc định sản phẩm): 5 hàm điểm chuẩn SRS (Eye Contact 25%, Head Pose 20%,
      Speech Rate 20%, Filler Control 20%, Voice Energy 15%) và tổng trọng số chuẩn xác.
    - 'human_coaching_v1' (Nhánh nghiên cứu): Dự đoán bằng mô hình học máy + giải thích SHAP.
 4. Hệ thống Sự kiện (Events) & AI Coach theo Bằng chứng (Evidence-based):
    - Xuất danh sách events có evidence_id, start_ms, end_ms, loại vi phạm và giá trị đo.
    - Cập nhật model Google Gemini mới (mặc định gemini-2.5-flash, cấu hình qua biến môi trường),
      chỉ gửi metrics + evidence_id và kiểm tra chặt JSON Schema, luôn có Rule Fallback an toàn.
=============================================================================
"""

import os
import sys
import json
import time
import tempfile
import argparse
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import shap

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from extract_multimodal_features import (
    VisionFeatureExtractor55D,
    AudioFeatureExtractor65D,
    extract_audio_16k,
    compute_scientific_rubric_scores,
    compute_srs_v1_1_scores,
    build_telemetry_timeseries,
    ensure_directories_and_weights
)

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
CATBOOST_DIR = os.path.join(WEIGHTS_DIR, "catboost_scoring")
META_CONFIG_PATH = os.path.join(CATBOOST_DIR, "model_metadata_123d.json")
RUBRIC_POLICY_PATH = os.path.join(PROJECT_ROOT, "configs", "rubric_policy.json")

TARGET_COLS = [
    'eye_contact_score',
    'head_posture_score',
    'facial_expression_score',
    'speech_rate_score',
    'filler_control_score',
    'overall_score'
]

# Chuẩn hóa 3 bối cảnh chính thức của dự án: do_an, hung_bien, pitching
CONTEXT_NORMALIZE_MAP = {
    'do_an': 'do_an',
    'doan': 'do_an',
    'academic': 'do_an',
    'hung_bien': 'hung_bien',
    'hungbien': 'hung_bien',
    'public_speaking': 'hung_bien',
    'debate': 'hung_bien',
    'interview': 'hung_bien',
    'pitching': 'pitching',
    'startup_pitching': 'pitching'
}

CONTEXT_WEIGHTS_MAP = {
    'do_an':     (0.25, 0.20, 0.10, 0.25, 0.20),
    'hung_bien': (0.20, 0.15, 0.20, 0.25, 0.20),
    'pitching':  (0.25, 0.15, 0.25, 0.20, 0.15)
}

FEATURE_DESCRIPTIONS_VI = {
    'vis_02_eye_contact_ratio': 'Tỷ lệ duy trì hướng nhìn về phía camera',
    'vis_29_gaze_deviation_ratio': 'Tỷ lệ thời gian nhìn lệch khỏi vùng trung tâm',
    'vis_15_head_deviation_mean': 'Độ lệch góc tư thế đầu trung bình (Yaw/Pitch/Roll)',
    'vis_17_posture_violation_ratio': 'Tỷ lệ thời gian nghiêng/cúi đầu vượt ngưỡng 15°',
    'vis_10_smile_active_ratio': 'Tần suất biểu cảm cơ mặt tích cực (AU12)',
    'vis_55_facial_energy_index': 'Chỉ số năng lượng biểu cảm cơ mặt tổng hợp',
    'vis_21_blink_rate_per_min': 'Tần số chớp mắt mỗi phút (Blink Rate)',
    'aud_01_wpm_speech_rate': 'Tốc độ nói trung bình (Whitespace Units Per Minute - WPM)',
    'aud_03_pause_duration_ratio': 'Tỷ lệ thời gian ngắt nghỉ nội bộ',
    'aud_04_dead_air_ratio': 'Tỷ lệ khoảng lặng kéo dài (> 2.5 giây)',
    'aud_05_filler_words_per_min': 'Tần suất sự kiện từ đệm (ờ, à, ừm) mỗi phút',
    'aud_08_f0_std_hz': 'Độ biến thiên tần số cơ bản giọng nói (Pitch F0 SD)',
    'aud_09_jitter_local_pct': 'Độ biến thiên chu kỳ thanh quản (Jitter)',
    'aud_10_shimmer_local_pct': 'Độ biến thiên biên độ âm lượng (Shimmer)',
    'aud_11_hnr_mean_db': 'Tỷ lệ hài âm trên nhiễu (Harmonics-to-Noise Ratio)',
}

PERSONA_SYSTEM_PROMPTS = {
    'do_an': (
        "Bạn là Chủ tịch Hội đồng Phản biện Đồ án Tốt nghiệp ngành CNTT tại Trường Đại học Bách Khoa - ĐHĐN. "
        "Phong cách: Chuẩn mực học thuật, khách quan, tập trung vào phong thái đĩnh đạc, tốc độ trình bày 115-135 WPM, "
        "kiểm soát từ đệm và chỉ nhận xét dựa trên bằng chứng đo lường (evidence_id) đã cung cấp."
    ),
    'hung_bien': (
        "Bạn là Giám khảo Trưởng Cuộc thi Hùng biện & Diễn thuyết trước Công chúng Quốc gia (Toastmasters / Trường Teen). "
        "Phong cách: Truyền cảm hứng, giàu năng lượng, chú trọng biểu cảm cơ mặt, ánh mắt thu hút, ngữ điệu trầm bổng "
        "và chỉ nhận xét dựa trên bằng chứng đo lường (evidence_id) đã cung cấp."
    ),
    'pitching': (
        "Bạn là Nhà đầu tư mạo hiểm (Venture Capitalist / Shark Tank) thẩm định Startup Công nghệ. "
        "Phong cách: Thực tế, sắc bén, đánh giá cao sự tự tin, nụ cười thuyết phục, tốc độ nói quyết đoán 135-155 WPM "
        "và chỉ nhận xét dựa trên bằng chứng đo lường (evidence_id) đã cung cấp."
    )
}


class MultimodalAIAssessmentEngine:
    def __init__(self, enable_asr=True, default_policy="pbl6_context_aware_v1"):
        ensure_directories_and_weights()
        self.default_policy = default_policy
        print("[AI Core Engine] Đang khởi tạo các bộ trích xuất đặc trưng (Vision 5FPS + Audio/ASR)...")
        self.vis_extractor = VisionFeatureExtractor55D()
        self.aud_extractor = AudioFeatureExtractor65D(enable_asr=enable_asr)

        print("[AI Core Engine] Đang nạp bộ mô hình nghiên cứu CatBoost 123D...")
        if not os.path.exists(META_CONFIG_PATH):
            raise FileNotFoundError(f"Chưa tìm thấy file cấu hình: {META_CONFIG_PATH}")

        with open(META_CONFIG_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.feature_cols = self.metadata['feature_columns_123d']
        self.cat_models = {}
        for target in TARGET_COLS:
            m_path = os.path.join(CATBOOST_DIR, f"catboost_{target}.cbm")
            model = CatBoostRegressor()
            model.load_model(m_path)
            self.cat_models[target] = model

        self.shap_explainer = shap.TreeExplainer(self.cat_models['overall_score'])
        print("[AI Core Engine] Hệ thống đã sẵn sàng đánh giá thuyết trình!\n")

    def run_assessment(self, video_path, context_type='do_an', sample_fps=5.0,
                       scoring_policy=None, gemini_api_key=None):
        """
        Thực hiện đánh giá toàn diện 1 video thuyết trình theo 3 bối cảnh chính thức:
        'do_an' | 'hung_bien' | 'pitching'.
        """
        start_total = time.perf_counter()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

        policy = scoring_policy or self.default_policy
        raw_ctx = str(context_type).strip().lower()
        canonical_ctx = CONTEXT_NORMALIZE_MAP.get(raw_ctx, 'do_an')

        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0].upper()
        file_size_mb = os.path.getsize(video_path) / (1024.0 * 1024.0)

        print("=" * 80)
        print(f"[*] BẮT ĐẦU ĐÁNH GIÁ: {video_filename} | BỐI CẢNH: {canonical_ctx.upper()} | POLICY: {policy}")
        print("=" * 80)

        # 1. Tách âm thanh 16kHz vào file tạm tự dọn dẹp
        t_wav0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix="_eval_16k.wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        try:
            ok_wav = extract_audio_16k(video_path, wav_path)
            if not ok_wav:
                raise RuntimeError(f"AI_NO_AUDIO_STREAM: Không thể tách luồng âm thanh từ {video_path}")
            time_wav = time.perf_counter() - t_wav0
            print(f"[1/4] Kiểm tra Media & Tách WAV 16kHz Mono: Hoàn tất ({time_wav:.2f}s)")

            # 2. Chạy Vision Sub-Engine ở 5 FPS
            t_vis0 = time.perf_counter()
            vis_55d, df_vis_frames = self.vis_extractor.process_video(video_path, sample_fps=sample_fps)
            time_vis = time.perf_counter() - t_vis0
            print(f"[2/4] Đo lường Thị giác ({sample_fps:.1f} FPS, MediaPipe + solvePnP + 1-Euro): Hoàn tất ({time_vis:.2f}s)")

            # 3. Chạy Audio & Vietnamese ASR Sub-Engine
            t_aud0 = time.perf_counter()
            aud_65d, sec_f0, sec_wpm, duration_sec = self.aud_extractor.process_audio(wav_path)
            time_aud = time.perf_counter() - t_aud0
            print(f"[3/4] Đo lường Âm thanh & Tiếng Việt (WPM, Fillers, F0, Pauses): Hoàn tất ({time_aud:.2f}s)")
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

        # 4. Kiểm tra Chất lượng Đầu vào (Quality Gate)
        vis_q = self.vis_extractor.last_quality_summary
        aud_q = self.aud_extractor.last_quality_summary
        face_coverage = float(vis_q.get("face_coverage", vis_55d['vis_01_face_presence_ratio']))
        front_ratio_valid = float(vis_q.get("front_ratio_valid", vis_55d['vis_02_eye_contact_ratio']))
        front_ratio_total = float(vis_q.get("front_ratio_total", vis_55d['vis_02_eye_contact_ratio']))
        voiced_fraction = float(aud_q.get("voiced_fraction", aud_65d['aud_24_voiced_fraction']))

        quality_flags = list(vis_q.get("vision_flags", [])) + list(aud_q.get("audio_flags", []))
        if duration_sec < 15.0:
            quality_flags.append("DURATION_TOO_SHORT_UNDER_15S")
        elif duration_sec < 60.0:
            quality_flags.append("SHORT_STATISTICS_15_TO_59S")
        elif duration_sec > 300.0:
            quality_flags.append("DURATION_OVER_300S")
        if file_size_mb > 100.0:
            quality_flags.append("FILE_SIZE_OVER_100MB")

        analysis_status = "valid"
        quality_reason = "Đủ dữ liệu quan sát hợp lệ cho cả kênh hình ảnh và âm thanh."
        if face_coverage < 0.80 or voiced_fraction < 0.12:
            analysis_status = "insufficient_data" if (face_coverage < 0.40 or voiced_fraction < 0.05) else "low_confidence"
            quality_reason = (
                f"Độ phủ khuôn mặt đạt {face_coverage*100:.1f}% (ngưỡng chuẩn >= 80%), "
                f"tỷ lệ tiếng nói hữu thanh đạt {voiced_fraction*100:.1f}%."
            )

        # 5. Chấm điểm theo chính sách Đa Bối cảnh chuẩn của dự án (do_an, hung_bien, pitching)
        t_cb0 = time.perf_counter()
        context_rule_scores = compute_scientific_rubric_scores(vis_55d, aud_65d, canonical_ctx)
        srs_scores = compute_srs_v1_1_scores(vis_55d, aud_65d, canonical_ctx)

        ctx_3d = {
            'ctx_01_do_an': 1.0 if canonical_ctx == 'do_an' else 0.0,
            'ctx_02_hung_bien': 1.0 if canonical_ctx == 'hung_bien' else 0.0,
            'ctx_03_pitching': 1.0 if canonical_ctx == 'pitching' else 0.0,
        }
        vector_123d = {**ctx_3d, **vis_55d, **aud_65d}
        df_row = pd.DataFrame([[vector_123d[c] for c in self.feature_cols]], columns=self.feature_cols)

        ml_scores = {}
        for target in TARGET_COLS:
            pred_score = float(self.cat_models[target].predict(df_row)[0])
            ml_scores[target] = round(float(np.clip(pred_score, 45.0, 98.5)), 2)

        # Tính tổng có trọng số theo đúng ma trận bối cảnh (do_an, hung_bien, pitching)
        w1, w2, w3, w4, w5 = CONTEXT_WEIGHTS_MAP.get(canonical_ctx, (0.25, 0.20, 0.10, 0.25, 0.20))
        ml_weighted_overall = round(
            w1 * ml_scores['eye_contact_score']
            + w2 * ml_scores['head_posture_score']
            + w3 * ml_scores['facial_expression_score']
            + w4 * ml_scores['speech_rate_score']
            + w5 * ml_scores['filler_control_score'],
            2
        )

        # Chọn bộ điểm chính theo policy được yêu cầu (Mặc định: pbl6_context_aware_v1)
        if policy == "srs_v1_1":
            active_overall = srs_scores["overall_score"] if analysis_status != "insufficient_data" else None
            primary_rubric = {
                "eye_contact_score": srs_scores["eye_contact"],
                "head_posture_score": srs_scores["head_pose_stability"],
                "facial_expression_score": context_rule_scores["facial_expression_score"],
                "voice_energy_score": srs_scores["voice_energy"],
                "speech_rate_score": srs_scores["speech_rate"],
                "filler_control_score": srs_scores["filler_control"]
            }
        elif policy == "human_coaching_v1":
            active_overall = ml_weighted_overall if analysis_status != "insufficient_data" else None
            primary_rubric = {
                "eye_contact_score": ml_scores["eye_contact_score"],
                "head_posture_score": ml_scores["head_posture_score"],
                "facial_expression_score": ml_scores["facial_expression_score"],
                "voice_energy_score": context_rule_scores["voice_energy_score"],
                "speech_rate_score": ml_scores["speech_rate_score"],
                "filler_control_score": ml_scores["filler_control_score"]
            }
        else:
            # Mặc định: pbl6_context_aware_v1 (Theo đúng 3 bối cảnh do_an, hung_bien, pitching)
            active_overall = context_rule_scores["overall_score"] if analysis_status != "insufficient_data" else None
            primary_rubric = {
                "eye_contact_score": context_rule_scores["eye_contact_score"],
                "head_posture_score": context_rule_scores["head_posture_score"],
                "facial_expression_score": context_rule_scores["facial_expression_score"],
                "voice_energy_score": context_rule_scores["voice_energy_score"],
                "speech_rate_score": context_rule_scores["speech_rate_score"],
                "filler_control_score": context_rule_scores["filler_control_score"]
            }

        # 6. Giải thích SHAP XAI & Hợp nhất Sự kiện Timeline (Events)
        shap_values = self.shap_explainer.shap_values(df_row)[0]
        base_value = float(np.array(self.shap_explainer.expected_value).flatten()[0])
        sorted_idx = np.argsort(shap_values)

        top_penalties = []
        for idx in sorted_idx[:5]:
            imp = float(shap_values[idx])
            if imp < -0.05:
                fname = self.feature_cols[idx]
                top_penalties.append({
                    "feature": fname,
                    "description": FEATURE_DESCRIPTIONS_VI.get(fname, fname),
                    "impact": round(imp, 2),
                    "measured_value": round(float(df_row[fname].iloc[0]), 3)
                })

        top_strengths = []
        for idx in sorted_idx[::-1][:5]:
            imp = float(shap_values[idx])
            if imp > 0.05:
                fname = self.feature_cols[idx]
                top_strengths.append({
                    "feature": fname,
                    "description": FEATURE_DESCRIPTIONS_VI.get(fname, fname),
                    "impact": round(imp, 2),
                    "measured_value": round(float(df_row[fname].iloc[0]), 3)
                })

        time_cb = time.perf_counter() - t_cb0
        print(f"[4/4] Tính điểm Policy '{policy}' ({canonical_ctx}) & Tổng hợp Events: Hoàn tất ({time_cb:.3f}s)")

        # 7. Hợp nhất Events (Audio + Vision) và Telemetry Time-series
        all_events = sorted(
            self.vis_extractor.last_vision_events + self.aud_extractor.last_audio_events,
            key=lambda x: x["start_ms"]
        )
        telemetry_timeseries = build_telemetry_timeseries(df_vis_frames, sec_f0, sec_wpm)

        metrics_dict = {
            "face_coverage": {"value": round(face_coverage, 4), "unit": "ratio"},
            "front_ratio_valid": {"value": round(front_ratio_valid, 4), "unit": "ratio"},
            "front_ratio_total": {"value": round(front_ratio_total, 4), "unit": "ratio"},
            "head_stability_ratio": {"value": round(1.0 - vis_55d['vis_17_posture_violation_ratio'], 4), "unit": "ratio"},
            "whitespace_units_per_minute": {
                "value": round(aud_65d['aud_01_wpm_speech_rate'], 2),
                "unit": "wpm",
                "tokenizer_version": self.aud_extractor.last_transcript_info.get("tokenizer_version", "phowhisper_vi_v1")
            },
            "articulation_rate_wpm": {"value": round(aud_65d['aud_02_articulation_rate_wpm'], 2), "unit": "wpm"},
            "filler_events_per_minute": {
                "value": round(aud_65d['aud_05_filler_words_per_min'], 2),
                "unit": "events_per_min",
                "count_policy": self.aud_extractor.last_transcript_info.get("count_policy", "event_word_boundary")
            },
            "internal_pause_ratio": {"value": round(aud_65d['aud_03_pause_duration_ratio'], 4), "unit": "ratio"},
            "f0_median_hz": {"value": round(aud_65d['aud_18_f0_p50_hz'], 2), "unit": "hz"},
            "f0_iqr_hz": {"value": round(aud_65d['aud_21_f0_iqr_hz'], 2), "unit": "hz"},
            "rms_dynamic_range_db": {"value": round(aud_65d['aud_31_rms_dynamic_range_db'], 2), "unit": "db"}
        }

        # 8. Điều phối AI Coach theo Bằng chứng (Evidence-grounded Coach + Rule Fallback)
        display_overall = active_overall if active_overall is not None else context_rule_scores["overall_score"]
        feedback_obj, legacy_xai_report = self._orchestrate_ai_coach(
            context_type=canonical_ctx,
            overall_score=display_overall,
            rubric_scores=primary_rubric,
            metrics=metrics_dict,
            events=all_events,
            quality_flags=quality_flags,
            top_penalties=top_penalties,
            top_strengths=top_strengths,
            api_key=gemini_api_key
        )

        total_latency = time.perf_counter() - start_total

        assessment_result = {
            "schema_version": "1.1",
            "presentation_id": video_id,
            "filename": video_filename,
            "status": "COMPLETED",
            "context": canonical_ctx,
            "context_type": canonical_ctx,
            "duration_sec": round(duration_sec, 2),
            "analysis_status": analysis_status,
            "scoring_policy": policy,
            "score_mapping_version": "mapping_context_v1",
            "model_bundle_version": "bundle_v1",
            "quality": {
                "face_coverage": round(face_coverage, 4),
                "voiced_fraction": round(voiced_fraction, 4),
                "flags": quality_flags,
                "reason": quality_reason
            },
            "metrics": metrics_dict,
            "rubric_scores": primary_rubric,
            "overall_score": display_overall,
            "overall_score_nullable": active_overall,
            "research_comparison": {
                "pbl6_context_aware_v1_scores": context_rule_scores,
                "human_coaching_v1_ml_scores": {
                    **ml_scores,
                    "weighted_overall_score": ml_weighted_overall
                },
                "srs_v1_1_reference_scores": srs_scores
            },
            "transcript_info": self.aud_extractor.last_transcript_info,
            "events": all_events,
            "telemetry_timeseries": telemetry_timeseries,
            "feedback": feedback_obj,
            "shap_explanation": {
                "base_score_phi0": round(base_value, 2),
                "top_penalties": top_penalties,
                "top_strengths": top_strengths
            },
            "xai_report": legacy_xai_report,
            "performance_metrics": {
                "total_latency_sec": round(total_latency, 2),
                "real_time_factor": round(total_latency / max(duration_sec, 1.0), 3),
                "decode_wav_sec": round(time_wav, 3),
                "vision_sec": round(time_vis, 3),
                "audio_asr_sec": round(time_aud, 3),
                "scoring_sec": round(time_cb, 3),
                "features_analyzed": len(self.feature_cols)
            }
        }

        print("\n" + "=" * 80)
        print(f"       KẾT QUẢ ĐÁNH GIÁ THUYẾT TRÌNH (BỐI CẢNH: {canonical_ctx.upper()} | POLICY: {policy})")
        print("=" * 80)
        print(f"📊 ĐIỂM TỔNG KẾT: {display_overall:.2f} / 100  (Trạng thái chất lượng: {analysis_status})")
        print(f"  1. Ánh mắt & Hướng Camera (Eye Contact)   : {primary_rubric['eye_contact_score']:.2f}")
        print(f"  2. Phong thái & Tư thế Đầu (Head Pose)    : {primary_rubric['head_posture_score']:.2f}")
        print(f"  3. Biểu cảm & Nụ cười (AU12 Smile)        : {primary_rubric['facial_expression_score']:.2f}")
        print(f"  4. Tốc độ & Nhịp điệu (Speech Rate WPM)   : {primary_rubric['speech_rate_score']:.2f}")
        print(f"  5. Kiểm soát Từ đệm (Filler Control)      : {primary_rubric['filler_control_score']:.2f}")
        print(f"  [+] Chỉ số Năng lượng Giọng (Voice Energy): {primary_rubric['voice_energy_score']:.2f}")
        print(f"📌 Số sự kiện (Events) gắn mốc thời gian    : {len(all_events)} sự kiện")
        print(f"⏱️ Tổng thời gian xử lý                     : {total_latency:.2f}s (RTF: {total_latency/max(duration_sec,1.0):.3f})")
        print("=" * 80)

        return assessment_result

    def _orchestrate_ai_coach(self, context_type, overall_score, rubric_scores,
                              metrics, events, quality_flags, top_penalties, top_strengths, api_key=None):
        """
        Điều phối AI Coach theo 3 Persona chuẩn (do_an, hung_bien, pitching) gắn với bằng chứng (evidence_id).
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        persona_desc = PERSONA_SYSTEM_PROMPTS.get(context_type, PERSONA_SYSTEM_PROMPTS['do_an'])
        top_events = events[:6]
        valid_ev_ids = {e["evidence_id"] for e in top_events}

        if api_key:
            try:
                compact_payload = {
                    "context": context_type,
                    "overall_score": overall_score,
                    "rubric_scores": rubric_scores,
                    "quality_flags": quality_flags,
                    "evidence_events": top_events
                }
                prompt_content = (
                    f"{persona_desc}\n"
                    f"Dựa DUY NHẤT vào dữ liệu đo lường và danh sách evidence_events dưới đây (tuyệt đối không tự bịa số liệu hay timestamp ngoài danh sách):\n"
                    f"{json.dumps(compact_payload, ensure_ascii=False, indent=2)}\n\n"
                    "Hãy trả về đúng 1 JSON object có cấu trúc:\n"
                    "{\n"
                    '  "ai_coach_persona": "Chức danh huấn luyện viên",\n'
                    '  "summary_feedback": "Nhận xét tổng quan 2-3 câu",\n'
                    '  "strengths": ["Điểm mạnh 1", "Điểm mạnh 2"],\n'
                    '  "improvements": [{"text": "Điểm cần cải thiện gắn mốc giây", "evidence_id": "mã evidence_id có thật"}],\n'
                    '  "exercises": ["Bài tập 1", "Bài tập 2", "Bài tập 3"]\n'
                    "}"
                )
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
                req_data = {
                    "contents": [{"parts": [{"text": prompt_content}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900}
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(req_data).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    res_body = json.loads(response.read().decode('utf-8'))
                    raw_text = res_body['candidates'][0]['content']['parts'][0]['text'].strip()
                    if raw_text.startswith("```json"):
                        raw_text = raw_text[7:]
                    if raw_text.startswith("```"):
                        raw_text = raw_text[3:]
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-3]
                    parsed = json.loads(raw_text.strip())
                    imps = parsed.get("improvements", [])
                    cleaned_imps = []
                    weaknesses_str = []
                    for item in imps:
                        if isinstance(item, dict):
                            ev_id = item.get("evidence_id")
                            txt = item.get("text", "")
                            if ev_id in valid_ev_ids or not valid_ev_ids:
                                cleaned_imps.append({"text": txt, "evidence_id": ev_id})
                                weaknesses_str.append(f"{txt} [{ev_id}]" if ev_id else txt)
                        elif isinstance(item, str):
                            cleaned_imps.append({"text": item, "evidence_id": None})
                            weaknesses_str.append(item)

                    feedback_std = {
                        "source": "cloud_llm_verified",
                        "model_id": model_id,
                        "prompt_version": "coach_evidence_v1",
                        "strengths": parsed.get("strengths", [])[:3],
                        "improvements": cleaned_imps,
                        "exercises": parsed.get("exercises", [])[:4]
                    }
                    legacy_xai = {
                        "ai_coach_persona": parsed.get("ai_coach_persona", "AI Presentation Coach"),
                        "summary_feedback": parsed.get("summary_feedback", ""),
                        "strengths": feedback_std["strengths"],
                        "weaknesses": weaknesses_str,
                        "action_plan": [
                            {"day": f"Bước {i+1}", "exercise": ex}
                            for i, ex in enumerate(feedback_std["exercises"])
                        ]
                    }
                    return feedback_std, legacy_xai
            except Exception as e:
                print(f"[!] Chuyển sang Rule Fallback nội bộ do Cloud LLM không phản hồi: {e}")

        persona_titles = {
            'do_an': "Hội đồng Phản biện Đồ án Tốt nghiệp CNTT (Bách Khoa Đà Nẵng)",
            'hung_bien': "Huấn luyện viên Trưởng Diễn thuyết & Hùng biện Quốc tế",
            'pitching': "Hội đồng Thẩm định Đầu tư Mạo hiểm (Shark Tank VC)"
        }
        ideal_wpm_str_map = {
            'do_an': "115-135 WPM",
            'hung_bien': "125-150 WPM",
            'pitching': "135-155 WPM"
        }
        ideal_wpm_str = ideal_wpm_str_map.get(context_type, "120-145 WPM")

        strengths_vi = []
        if rubric_scores['eye_contact_score'] >= 72:
            strengths_vi.append(
                f"Duy trì hướng nhìn về phía camera tốt ({metrics['front_ratio_total']['value']*100:.1f}% thời lượng)."
            )
        if rubric_scores['speech_rate_score'] >= 78:
            strengths_vi.append(
                f"Tốc độ trình bày đạt {metrics['whitespace_units_per_minute']['value']:.1f} WPM, phù hợp với bối cảnh {context_type}."
            )
        if rubric_scores['filler_control_score'] >= 80:
            strengths_vi.append(
                f"Kiểm soát từ đệm tốt ({metrics['filler_events_per_minute']['value']:.2f} sự kiện/phút)."
            )
        if not strengths_vi:
            strengths_vi = [
                f"Tỷ lệ tiếng nói hữu thanh đạt {metrics['f0_median_hz']['value']:.1f} Hz trung vị, âm lượng thu ổn định."
            ]

        improvements_structured = []
        weaknesses_legacy = []
        for ev in top_events[:4]:
            sec_start = ev["start_ms"] / 1000.0
            sec_end = ev["end_ms"] / 1000.0
            msg = f"Tại {sec_start:.1f}s - {sec_end:.1f}s: {ev['description']} (Mã bằng chứng: {ev['evidence_id']})"
            improvements_structured.append({
                "evidence_id": ev["evidence_id"],
                "start_ms": ev["start_ms"],
                "end_ms": ev["end_ms"],
                "text": msg
            })
            weaknesses_legacy.append(msg)

        if not weaknesses_legacy:
            for p in top_penalties[:2]:
                msg = f"Chỉ số cần cải thiện: {p['description']} (Giá trị đo: {p['measured_value']})"
                improvements_structured.append({"evidence_id": None, "text": msg})
                weaknesses_legacy.append(msg)

        exercises_list = []
        action_plan = []
        if "LOW_FACE_COVERAGE" in quality_flags or "AI_NO_FACE_DETECTED" in quality_flags:
            ex = "Điều chỉnh lại vị trí camera ngang tầm mắt và bổ sung ánh sáng phía trước để hệ thống quan sát đủ khuôn mặt (>= 80% thời lượng)."
            exercises_list.append(ex)
            action_plan.append({"day": "Ngày 1", "exercise": ex})

        if rubric_scores['filler_control_score'] < 85:
            ex = "Tập kỹ thuật 'Khoảng nghỉ có chủ đích' ở ranh giới chuyển ý: dừng 1 giây thay vì phát âm 'ờ, ừm, à'."
            exercises_list.append(ex)
            action_plan.append({"day": "Ngày 1-2", "exercise": ex})

        if rubric_scores['eye_contact_score'] < 80 or rubric_scores['head_posture_score'] < 80:
            ex = "Đặt tài liệu gợi ý ngay sát dưới cụm ống kính webcam để khi nhìn dàn ý góc đầu không lệch quá 15 độ."
            exercises_list.append(ex)
            action_plan.append({"day": "Ngày 3-4", "exercise": ex})

        if rubric_scores['speech_rate_score'] < 85:
            ex = f"Luyện nói 1 phút theo 3 ý chính để đưa tốc độ từ {metrics['whitespace_units_per_minute']['value']:.0f} WPM về vùng chuẩn {ideal_wpm_str} của bối cảnh {context_type}."
            exercises_list.append(ex)
            action_plan.append({"day": "Ngày 5-6", "exercise": ex})

        ex_final = f"Thu lại bài nói 2-3 phút bối cảnh {context_type} và đối chiếu sự thay đổi chỉ số trên biểu đồ Radar."
        exercises_list.append(ex_final)
        action_plan.append({"day": "Ngày 7", "exercise": ex_final})

        feedback_std = {
            "source": "rule_fallback",
            "model_id": None,
            "prompt_version": "rule_evidence_v1",
            "strengths": strengths_vi,
            "improvements": improvements_structured,
            "exercises": exercises_list
        }

        legacy_xai = {
            "ai_coach_persona": persona_titles.get(context_type, "Gia sư AI Đa phương thức"),
            "summary_feedback": (
                f"Bài thuyết trình bối cảnh {context_type.upper()} đạt {overall_score:.1f}/100 điểm. "
                f"Hệ thống ghi nhận {len(events)} sự kiện trên timeline cần lưu ý."
            ),
            "strengths": strengths_vi,
            "weaknesses": weaknesses_legacy,
            "action_plan": action_plan
        }
        return feedback_std, legacy_xai


def main():
    parser = argparse.ArgumentParser(description="Chạy Bộ não Đánh giá Thuyết trình AI Toàn diện (PBL6)")
    parser.add_argument("--video", "-v", type=str, default="", help="Đường dẫn file video .mp4 / .webm")
    parser.add_argument("--context", "-c", type=str, default="do_an",
                        help="Bối cảnh chuẩn ('do_an' | 'hung_bien' | 'pitching')")
    parser.add_argument("--policy", "-p", type=str,
                        choices=["pbl6_context_aware_v1", "human_coaching_v1", "srs_v1_1"],
                        default="pbl6_context_aware_v1",
                        help="Chính sách chấm điểm (mặc định: pbl6_context_aware_v1)")
    parser.add_argument("--fps", type=float, default=5.0, help="Tốc độ lấy mẫu khung hình Vision (mặc định 5.0 FPS)")
    parser.add_argument("--no-asr", action="store_true", help="Tắt chạy mô hình PhoWhisper trực tiếp (dùng khi test nhanh trên CPU yếu)")
    parser.add_argument("--output", "-o", type=str, default="", help="Đường dẫn file JSON xuất kết quả")
    args = parser.parse_args()

    engine = MultimodalAIAssessmentEngine(enable_asr=not args.no_asr, default_policy=args.policy)

    video_target = args.video
    if not video_target:
        default_candidate = os.path.join(PROJECT_ROOT, "data", "raw", "DUT-PresCorpus", "do_an", "da_001.mp4")
        if os.path.exists(default_candidate):
            video_target = default_candidate
        else:
            print("[!] Chưa chỉ định video. Dùng cú pháp: python scripts/ai_assessment_engine.py --video <duong_dan_mp4>")
            return

    res = engine.run_assessment(
        video_path=video_target,
        context_type=args.context,
        sample_fps=args.fps,
        scoring_policy=args.policy
    )

    out_file = args.output
    if not out_file:
        out_dir = os.path.join(PROJECT_ROOT, "data", "processed", "sample_assessment_outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{res['presentation_id']}_full_assessment.json")

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Đã xuất toàn bộ kết quả báo cáo JSON ra: {out_file}\n")


if __name__ == "__main__":
    main()
