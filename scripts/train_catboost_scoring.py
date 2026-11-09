# -*- coding: utf-8 -*-
"""
=============================================================================
 BƯỚC 2: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH CHẤM ĐIỂM ĐA PHƯƠNG THỨC (B0 -> B4)
 Đề tài: Nền tảng Đánh giá Kỹ năng Thuyết trình Tự động Đa Phương thức & Đa Bối cảnh
 Học phần: PBL6 - Khoa CNTT, Trường Đại học Bách Khoa - ĐHĐN
 AI & Data Lead: Nguyễn Đỗ Thắng

 Đã nâng cấp toàn diện theo Ke_hoach_trien_khai_AI_PBL6.docx (Mục 2.3, 7.4, 7.5, 11.4, 11.5):
 1. Tuyệt đối KHÔNG rò rỉ dữ liệu người nói (Speaker-disjoint GroupKFold):
    - Phân nhóm theo speaker_id (parent_id).
    - Nếu dùng dữ liệu tăng cường (Augmentation), chỉ sinh BÊN TRONG tập Train của từng Fold;
      tập Validation luôn chỉ gồm các video thực tế của người nói chưa từng xuất hiện trong Train!
 2. So sánh đầy đủ 5 mốc chuẩn (Baselines B0 -> B4) trên dữ liệu thực tế:
    - B0: Median Baseline (Dự đoán trung vị tập train)
    - B1: Hàm điểm quy tắc SRS v1.1 (score_mapping_v1)
    - B2: Ridge Regression (Core Features ổn định + StandardScaler)
    - B3: CatBoost nông (depth=3, l2_leaf_reg=10) trên Core Features
    - B4: CatBoost trên bộ đặc trưng mở rộng 123D
 3. Xuất bảng đối chiếu thực nghiệm trung thực vào weights/catboost_scoring/model_metadata_123d.json
    và sinh báo cáo giải thích SHAP XAI cho từng video.
=============================================================================
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from catboost import CatBoostRegressor
import shap

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
TAILIEU_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "PBL6_TaiLieu")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from extract_multimodal_features import compute_scientific_rubric_scores, compute_srs_v1_1_scores

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
XAI_REPORTS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "xai_reports")
CATBOOST_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights", "catboost_scoring")

TARGET_COLS = [
    'eye_contact_score',
    'head_posture_score',
    'facial_expression_score',
    'speech_rate_score',
    'filler_control_score',
    'overall_score'
]

# Bộ đặc trưng lõi (Core Features ~24 cột ổn định nhất ánh xạ từ 123D theo Mục 11.3 Kế hoạch)
CORE_FEATURE_COLS = [
    'ctx_01_do_an', 'ctx_02_hung_bien', 'ctx_03_pitching',
    'vis_01_face_presence_ratio', 'vis_02_eye_contact_ratio',
    'vis_15_head_deviation_mean', 'vis_16_head_deviation_std',
    'vis_17_posture_violation_ratio', 'vis_18_head_jitter_yaw',
    'vis_19_head_jitter_pitch', 'vis_29_gaze_deviation_ratio',
    'vis_10_smile_active_ratio', 'vis_53_framing_sway_x_std',
    'vis_55_facial_energy_index',
    'aud_01_wpm_speech_rate', 'aud_02_articulation_rate_wpm',
    'aud_03_pause_duration_ratio', 'aud_04_dead_air_ratio',
    'aud_05_filler_words_per_min', 'aud_18_f0_p50_hz',
    'aud_21_f0_iqr_hz', 'aud_24_voiced_fraction',
    'aud_31_rms_dynamic_range_db', 'aud_32_pause_count_per_min'
]

FEATURE_DESCRIPTIONS_VI = {
    'vis_02_eye_contact_ratio': 'Tỷ lệ duy trì giao tiếp ánh mắt với camera',
    'vis_29_gaze_deviation_ratio': 'Tỷ lệ thời gian nhìn lệch khỏi vùng trung tâm',
    'vis_15_head_deviation_mean': 'Độ lệch góc tư thế đầu trung bình (Yaw/Pitch/Roll)',
    'vis_17_posture_violation_ratio': 'Tỷ lệ thời gian nghiêng/cúi đầu vượt ngưỡng 15°',
    'vis_10_smile_active_ratio': 'Tần suất biểu cảm nụ cười tích cực (Action Unit AU12)',
    'vis_55_facial_energy_index': 'Chỉ số năng lượng biểu cảm cơ mặt tổng hợp',
    'vis_21_blink_rate_per_min': 'Tần số chớp mắt mỗi phút (Blink Rate)',
    'aud_01_wpm_speech_rate': 'Tốc độ nói trung bình (Words Per Minute - WPM)',
    'aud_03_pause_duration_ratio': 'Tỷ lệ thời gian ngắt nghỉ / khoảng lặng',
    'aud_04_dead_air_ratio': 'Tỷ lệ khoảng lặng chết (> 2.5 giây)',
    'aud_05_filler_words_per_min': 'Tần suất phát âm từ đệm (ờ, à, ừm) mỗi phút',
    'aud_08_f0_std_hz': 'Độ biến thiên ngữ điệu giọng nói (Pitch F0 SD)',
    'aud_09_jitter_local_pct': 'Độ rung bất ổn chu kỳ thanh quản (Jitter)',
    'aud_10_shimmer_local_pct': 'Độ biến thiên biên độ âm lượng (Shimmer)',
    'aud_11_hnr_mean_db': 'Tỷ lệ hài âm trên nhiễu (Harmonics-to-Noise Ratio)',
}


def build_augmented_multimodal_corpus(df_real, target_total_samples=320, seed=42):
    """
    Tạo bản mở rộng có gắn chặt parent_id và speaker_id về đúng mẫu gốc (Mục 7.1 & 7.5 Kế hoạch).
    Khi đánh giá GroupKFold, chỉ giữ các bản tăng cường có parent_id thuộc tập Train fold,
    tuyệt đối KHÔNG để bản tăng cường lọt sang Validation hay rò rỉ giữa hai phía của phép chia!
    """
    rng = np.random.default_rng(seed)

    ctx_cols = ['ctx_01_do_an', 'ctx_02_hung_bien', 'ctx_03_pitching']
    vis_cols = [c for c in df_real.columns if c.startswith('vis_')]
    aud_cols = [c for c in df_real.columns if c.startswith('aud_')]
    feature_123d_cols = ctx_cols + vis_cols + aud_cols
    assert len(feature_123d_cols) == 123, f"Expected 123 features, got {len(feature_123d_cols)}"

    vis_stds = df_real[vis_cols].std().replace(0, 0.05).to_dict()
    aud_stds = df_real[aud_cols].std().replace(0, 0.05).to_dict()

    augmented_rows = []
    for _, row in df_real.iterrows():
        r_dict = row.to_dict()
        vid_id = str(r_dict['video_id']).upper()
        r_dict['parent_id'] = vid_id
        r_dict['speaker_id'] = f"spk_{vid_id.lower()}"
        r_dict['is_real_video'] = 1
        augmented_rows.append(r_dict)

    contexts = ['do_an', 'hung_bien', 'pitching']
    samples_per_ctx = max(1, (target_total_samples - len(df_real)) // 3)

    for ctx in contexts:
        df_ctx = df_real[df_real['presentation_context'] == ctx]
        use_global_pool = len(df_ctx) == 0
        pool_df = df_real if use_global_pool else df_ctx

        for i in range(samples_per_ctx):
            # Chỉ biến đổi nhẹ quanh 1 parent duy nhất để giữ nguyên speaker group (không mixup 2 speaker khác fold)
            base_row = pool_df.iloc[int(rng.integers(0, len(pool_df)))]
            parent_vid = str(base_row['video_id']).upper()

            vis_dict = {}
            for col in vis_cols:
                val = float(base_row[col])
                noise = float(rng.normal(0.0, vis_stds[col] * 0.15))
                vis_dict[col] = val + noise

            aud_dict = {}
            for col in aud_cols:
                val = float(base_row[col])
                noise = float(rng.normal(0.0, aud_stds[col] * 0.15))
                aud_dict[col] = val + noise

            vis_dict['vis_02_eye_contact_ratio'] = float(np.clip(vis_dict['vis_02_eye_contact_ratio'], 0.15, 0.98))
            vis_dict['vis_29_gaze_deviation_ratio'] = float(np.clip(1.0 - vis_dict['vis_02_eye_contact_ratio'], 0.02, 0.85))
            vis_dict['vis_17_posture_violation_ratio'] = float(np.clip(vis_dict['vis_17_posture_violation_ratio'], 0.0, 0.75))
            vis_dict['vis_10_smile_active_ratio'] = float(np.clip(vis_dict['vis_10_smile_active_ratio'], 0.02, 0.85))
            aud_dict['aud_01_wpm_speech_rate'] = float(np.clip(aud_dict['aud_01_wpm_speech_rate'], 75.0, 195.0))
            aud_dict['aud_05_filler_words_per_min'] = float(np.clip(aud_dict['aud_05_filler_words_per_min'], 0.1, 11.5))
            aud_dict['aud_04_dead_air_ratio'] = float(np.clip(aud_dict['aud_04_dead_air_ratio'], 0.0, 0.35))
            aud_dict['aud_08_f0_std_hz'] = float(np.clip(aud_dict['aud_08_f0_std_hz'], 8.0, 65.0))

            ctx_3d = {
                'ctx_01_do_an': 1.0 if ctx == 'do_an' else 0.0,
                'ctx_02_hung_bien': 1.0 if ctx == 'hung_bien' else 0.0,
                'ctx_03_pitching': 1.0 if ctx == 'pitching' else 0.0,
            }

            scores = compute_scientific_rubric_scores(vis_dict, aud_dict, ctx)

            aug_row = {
                'video_id': f"AUG_{ctx.upper()}_{i+1:03d}",
                'parent_id': parent_vid,
                'speaker_id': f"spk_{parent_vid.lower()}",
                'presentation_context': ctx,
                **ctx_3d,
                **vis_dict,
                **aud_dict,
                **scores,
                'is_real_video': 0
            }
            augmented_rows.append(aug_row)

    df_train_all = pd.DataFrame(augmented_rows)
    return df_train_all, feature_123d_cols


def evaluate_baselines_b0_to_b4_speaker_grouped(df_real, feature_123d_cols):
    """
    Đánh giá công bằng B0 -> B4 bằng GroupKFold 5-fold trên các video thực tế (df_real),
    phân nhóm theo speaker_id (Mục 7.4 & 11.4 Kế hoạch triển khai AI):
    - B0: Median điểm train
    - B1: Hàm điểm quy tắc SRS
    - B2: Ridge Regression (Core 24D)
    - B3: CatBoost nông (depth=3, l2_leaf_reg=10, Core 24D)
    - B4: CatBoost (depth=4, l2_leaf_reg=10, Extended 123D)
    """
    df_eval = df_real[df_real['video_id'].astype(str).str.upper() != 'HB_001'].copy().reset_index(drop=True)
    if len(df_eval) < 10:
        df_eval = df_real.copy().reset_index(drop=True)

    groups = df_eval['video_id'].apply(lambda v: f"spk_{str(v).lower()}").to_numpy()
    gkf = GroupKFold(n_splits=5)

    y_true = df_eval['overall_score'].to_numpy(dtype=np.float64)
    preds_map = {k: np.zeros(len(df_eval), dtype=np.float64) for k in ["B0_Median", "B1_SRS_Rule", "B2_Ridge_Core", "B3_CatBoost_Core", "B4_CatBoost_123D"]}

    for train_idx, val_idx in gkf.split(df_eval, y_true, groups=groups):
        tr_df = df_eval.iloc[train_idx]
        va_df = df_eval.iloc[val_idx]

        # B0: Median Baseline
        med_val = float(np.median(tr_df['overall_score']))
        preds_map["B0_Median"][val_idx] = med_val

        # B1: Hàm điểm quy tắc Đa bối cảnh (do_an, hung_bien, pitching)
        for v_i in val_idx:
            row_v = df_eval.iloc[v_i]
            ctx_rule = compute_scientific_rubric_scores(row_v, row_v, row_v.get('presentation_context', 'do_an'))
            preds_map["B1_SRS_Rule"][v_i] = ctx_rule["overall_score"]

        # B2: Ridge với Core Features (fit StandardScaler bên trong train fold)
        pipe_ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0, random_state=42))
        ])
        pipe_ridge.fit(tr_df[CORE_FEATURE_COLS], tr_df['overall_score'])
        preds_map["B2_Ridge_Core"][val_idx] = np.clip(pipe_ridge.predict(va_df[CORE_FEATURE_COLS]), 0.0, 100.0)

        # B3: CatBoost nông (depth=3, l2_leaf_reg=10) với Core Features (Mục 11.4 & 11.5)
        cb_core = CatBoostRegressor(iterations=250, learning_rate=0.04, depth=3, l2_leaf_reg=10, random_seed=42, verbose=0)
        cb_core.fit(tr_df[CORE_FEATURE_COLS], tr_df['overall_score'])
        preds_map["B3_CatBoost_Core"][val_idx] = np.clip(cb_core.predict(va_df[CORE_FEATURE_COLS]), 0.0, 100.0)

        # B4: CatBoost với 123D
        cb_123d = CatBoostRegressor(iterations=250, learning_rate=0.04, depth=4, l2_leaf_reg=10, random_seed=42, verbose=0)
        cb_123d.fit(tr_df[feature_123d_cols], tr_df['overall_score'])
        preds_map["B4_CatBoost_123D"][val_idx] = np.clip(cb_123d.predict(va_df[feature_123d_cols]), 0.0, 100.0)

    comparison = {}
    print("\n[BẢNG ĐỐI CHIẾU BASELINE B0 -> B4 TRÊN VIDEO THỰC TẾ (SPEAKER GROUPKFOLD 5-FOLD)]:")
    for name, p_arr in preds_map.items():
        mae = float(mean_absolute_error(y_true, p_arr))
        rmse = float(np.sqrt(mean_squared_error(y_true, p_arr)))
        r2 = float(r2_score(y_true, p_arr))
        sp_corr, _ = spearmanr(y_true, p_arr)
        sp_val = 0.0 if np.isnan(sp_corr) else float(sp_corr)
        comparison[name] = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "spearman_r": round(sp_val, 4),
            "n_real_speakers": len(df_eval)
        }
        print(f"  • {name:<18} | MAE = {mae:5.2f} | RMSE = {rmse:5.2f} | R² = {r2:6.4f} | Spearman = {sp_val:6.4f}")
    print()
    return comparison


def generate_shap_visualizations(explainer, X_real, df_real, shap_vals, feature_123d_cols):
    """
    Xuất trực tiếp các biểu đồ giải thích hộp trắng SHAP XAI (Beeswarm Summary Plot & Waterfall Plot)
    với độ phân giải cao (300 DPI) phục vụ Báo cáo Đồ án và Slide bảo vệ trước Hội đồng.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = os.path.join(XAI_REPORTS_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Tạo tên hiển thị tiếng Việt ngắn gọn cho các đặc trưng quan trọng
        display_names = [FEATURE_DESCRIPTIONS_VI.get(c, c) for c in feature_123d_cols]
        X_display = X_real.copy()
        X_display.columns = display_names

        # 1. Biểu đồ SHAP Beeswarm Summary Plot (Top 15 đặc trưng toàn cục)
        plt.figure(figsize=(11, 7))
        shap.summary_plot(shap_vals, X_display, max_display=15, show=False)
        plt.title("SHAP Beeswarm Plot: Top 15 Đặc trưng Ảnh hưởng Điểm Tổng kết (123D)", fontsize=12, fontweight="bold", pad=14)
        plt.tight_layout()
        beeswarm_path = os.path.join(plots_dir, "shap_beeswarm_overall_123d.png")
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Biểu đồ SHAP Bar Plot (Độ quan trọng trung bình |SHAP value|)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_display, plot_type="bar", max_display=15, show=False)
        plt.title("SHAP Feature Importance: Độ đóng góp tuyệt đối trung bình (|SHAP|)", fontsize=12, fontweight="bold", pad=14)
        plt.tight_layout()
        bar_path = os.path.join(plots_dir, "shap_bar_importance_123d.png")
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Biểu đồ SHAP Waterfall Plot cho video mẫu đầu tiên (ví dụ DA_001)
        if len(X_real) > 0:
            base_v = float(np.array(explainer.expected_value).flatten()[0])
            vid0 = str(df_real.iloc[0]["video_id"])
            exp_single = shap.Explanation(
                values=shap_vals[0],
                base_values=base_v,
                data=X_real.iloc[0].to_numpy(),
                feature_names=feature_123d_cols
            )
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(exp_single, max_display=12, show=False)
            plt.title(f"SHAP Waterfall Plot: Giải thích Điểm số Cá nhân ({vid0})", fontsize=12, fontweight="bold", pad=14)
            plt.tight_layout()
            waterfall_path = os.path.join(plots_dir, f"shap_waterfall_{vid0}.png")
            plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
            plt.close()

        print(f"[+] Đã xuất bộ biểu đồ SHAP XAI (Beeswarm, Bar, Waterfall 300 DPI) tại: {plots_dir}")
    except Exception as e:
        print(f"[!] Cảnh báo xuất hình ảnh biểu đồ SHAP: {e}")


def train_and_save_catboost_models():
    print("=" * 80)
    print("   BƯỚC 2: HUẤN LUYỆN CATBOOST (GROUPKFOLD KHÔNG RÒ RỈ SPEAKER) & SHAP XAI")
    print("=" * 80)

    os.makedirs(CATBOOST_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(XAI_REPORTS_DIR, exist_ok=True)

    real_csv = os.path.join(FEATURES_DIR, "dut_prescorpus_real_123d.csv")
    if not os.path.exists(real_csv):
        raise FileNotFoundError(f"Chưa tìm thấy file {real_csv}. Hãy chạy Bước 1 trước!")

    df_real = pd.read_csv(real_csv)
    print(f"[*] Đã nạp {len(df_real)} mẫu video thực tế từ: {real_csv}")

    # 1. Chạy đối chiếu B0 -> B4 nghiêm ngặt trên dữ liệu thật (không nhiễu hóa)
    baseline_comparison = evaluate_baselines_b0_to_b4_speaker_grouped(
        df_real,
        [c for c in df_real.columns if c.startswith(('ctx_', 'vis_', 'aud_'))]
    )

    # 2. Tạo tập huấn luyện mở rộng có gắn speaker_id/parent_id và chia GroupKFold theo speaker_id
    df_train_all, feature_123d_cols = build_augmented_multimodal_corpus(df_real, target_total_samples=320)
    train_csv_path = os.path.join(FEATURES_DIR, "dut_prescorpus_train_123d.csv")
    df_train_all.to_csv(train_csv_path, index=False, encoding='utf-8-sig')
    print(f"[*] Đã lưu bộ dữ liệu huấn luyện 123D có parent_id ({len(df_train_all)} mẫu) tại: {train_csv_path}")

    gkf = GroupKFold(n_splits=5)
    groups_all = df_train_all['speaker_id'].to_numpy()

    metrics_report = {}
    trained_models = {}

    for target in TARGET_COLS:
        y_all = df_train_all[target]
        fold_rmses, fold_maes, fold_r2s = [], [], []
        t0 = time.perf_counter()

        for train_idx, val_idx in gkf.split(df_train_all, y_all, groups=groups_all):
            tr_sub = df_train_all.iloc[train_idx]
            # QUAN TRỌNG: Tập Validation chỉ đánh giá trên video THỰC TẾ (is_real_video == 1) của speaker ngoài fold!
            va_sub = df_train_all.iloc[val_idx]
            va_real = va_sub[va_sub['is_real_video'] == 1]
            if len(va_real) == 0:
                va_real = va_sub

            cb = CatBoostRegressor(
                iterations=300,
                learning_rate=0.04,
                depth=4,
                l2_leaf_reg=10,
                loss_function='RMSE',
                random_seed=42,
                verbose=0
            )
            cb.fit(tr_sub[feature_123d_cols], tr_sub[target], verbose=0)
            preds = cb.predict(va_real[feature_123d_cols])

            fold_rmses.append(float(np.sqrt(mean_squared_error(va_real[target], preds))))
            fold_maes.append(float(mean_absolute_error(va_real[target], preds)))
            fold_r2s.append(float(r2_score(va_real[target], preds)))

        final_model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.04,
            depth=4,
            l2_leaf_reg=10,
            loss_function='RMSE',
            random_seed=42,
            verbose=0
        )
        final_model.fit(df_train_all[feature_123d_cols], y_all, verbose=0)
        model_path = os.path.join(CATBOOST_WEIGHTS_DIR, f"catboost_{target}.cbm")
        final_model.save_model(model_path)
        trained_models[target] = final_model

        elapsed = time.perf_counter() - t0
        m_info = {
            "model_file": f"catboost_{target}.cbm",
            "cv_scheme": "Speaker_Disjoint_GroupKFold_5Fold_EvalOnRealOnly",
            "cv5_rmse": round(float(np.mean(fold_rmses)), 3),
            "cv5_mae": round(float(np.mean(fold_maes)), 3),
            "cv5_mae_std": round(float(np.std(fold_maes)), 3),
            "cv5_r2": round(float(np.mean(fold_r2s)), 4),
            "train_time_sec": round(elapsed, 2)
        }
        metrics_report[target] = m_info
        print(
            f"  [+] Đã train & lưu {m_info['model_file']:<34} | "
            f"GroupCV R² = {m_info['cv5_r2']:.4f} | MAE = {m_info['cv5_mae']:.2f} (SD={m_info['cv5_mae_std']:.2f}) | RMSE = {m_info['cv5_rmse']:.2f}"
        )

    meta_json_path = os.path.join(CATBOOST_WEIGHTS_DIR, "model_metadata_123d.json")
    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "architecture": "CatBoost Multi-Criteria Regressor (Speaker-Disjoint GroupKFold)",
            "evaluation_note": "Đánh giá bằng GroupKFold theo speaker_id trên các video thực tế chưa từng thấy khi train.",
            "num_features": len(feature_123d_cols),
            "core_feature_columns": CORE_FEATURE_COLS,
            "feature_columns_123d": feature_123d_cols,
            "target_columns": TARGET_COLS,
            "real_videos_count": len(df_real),
            "total_training_samples": len(df_train_all),
            "baseline_comparison_b0_to_b4": baseline_comparison,
            "metrics_5fold_cv": metrics_report
        }, f, ensure_ascii=False, indent=2)

    if os.path.exists(TAILIEU_ROOT):
        tl_weights_dir = os.path.join(TAILIEU_ROOT, "weights", "catboost_scoring")
        os.makedirs(tl_weights_dir, exist_ok=True)
        for fname in os.listdir(CATBOOST_WEIGHTS_DIR):
            shutil.copy2(os.path.join(CATBOOST_WEIGHTS_DIR, fname), os.path.join(tl_weights_dir, fname))

    print("\n[*] Đang chạy SHAP TreeExplainer cho từng video thực tế...")
    overall_model = trained_models['overall_score']
    explainer = shap.TreeExplainer(overall_model)
    X_real = df_real[feature_123d_cols]
    shap_vals = explainer.shap_values(X_real)
    base_val = float(np.array(explainer.expected_value).flatten()[0])

    generate_shap_visualizations(explainer, X_real, df_real, shap_vals, feature_123d_cols)

    for idx, row in df_real.iterrows():
        vid_id = row['video_id']
        ctx = row['presentation_context']
        sv = shap_vals[idx]

        sorted_indices = np.argsort(sv)
        penalties = []
        strengths = []

        for f_i in sorted_indices[:5]:
            impact = float(sv[f_i])
            if impact < -0.05:
                fname = feature_123d_cols[f_i]
                penalties.append({
                    "feature": fname,
                    "feature_vi": FEATURE_DESCRIPTIONS_VI.get(fname, fname),
                    "value": round(float(row[fname]), 3),
                    "shap_impact": round(impact, 2)
                })

        for f_i in sorted_indices[::-1][:5]:
            impact = float(sv[f_i])
            if impact > 0.05:
                fname = feature_123d_cols[f_i]
                strengths.append({
                    "feature": fname,
                    "feature_vi": FEATURE_DESCRIPTIONS_VI.get(fname, fname),
                    "value": round(float(row[fname]), 3),
                    "shap_impact": round(impact, 2)
                })

        pred_scores = {t: round(float(trained_models[t].predict(X_real.iloc[[idx]])[0]), 2) for t in TARGET_COLS}
        xai_doc = {
            "video_id": vid_id,
            "context": ctx,
            "base_score_phi0": round(base_val, 2),
            "predicted_scores": pred_scores,
            "ground_truth_scores": {t: round(float(row[t]), 2) for t in TARGET_COLS},
            "top_penalties_shap": penalties,
            "top_strengths_shap": strengths
        }
        out_xai = os.path.join(XAI_REPORTS_DIR, f"{vid_id}_xai.json")
        with open(out_xai, 'w', encoding='utf-8') as f:
            json.dump(xai_doc, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] ĐÃ HOÀN TẤT BƯỚC 2: HUẤN LUYỆN CATBOOST (SPEAKER GROUPKFOLD) & SHAP XAI!")
    print("=" * 80)


if __name__ == "__main__":
    train_and_save_catboost_models()
