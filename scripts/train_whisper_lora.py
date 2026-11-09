# -*- coding: utf-8 -*-
"""
=============================================================================
 CHƯƠNG TRÌNH HUẤN LUYỆN TINH CHỈNH PHOWHISPER VỚI KIẾN TRÚC LORA MỞ RỘNG
 Đề tài: Nền tảng Đánh giá Kỹ năng Thuyết trình Tự động (PBL6 - ĐHBK Đà Nẵng)
 AI & Data Lead: Nguyễn Đỗ Thắng
 Giảng viên hướng dẫn: TS. Lê Minh Trí

 CẢI TIẾN KHOA HỌC & KIẾN TRÚC MÔ HÌNH (ARCHITECTURAL ENHANCEMENT):
 1. Base Model: vinai/PhoWhisper-small (Pre-trained 840h tiếng Việt)
 2. Chuẩn hóa Phương ngữ & Dữ liệu (Theo góp ý của GVHD):
    - Tập trung vào Giọng chuẩn miền Nam / Phổ thông (VIVOS Corpus + Thuyết trình chuẩn)
      nhằm loại bỏ nhiễu biến âm phương ngữ miền Trung quá phức tạp gây Underfitting.
    - Chia tập Speaker-disjoint (không rò rỉ giọng người nói giữa Train / Dev / Test).
 3. Can thiệp Kiến trúc LoRA Mở rộng (Extended Cross-Attention & FFN LoRA):
    - Không chỉ gắn LoRA ở ["q_proj", "v_proj"] đơn thuần mà mở rộng sang:
      + Các tầng Cross-Attention của Decoder (encoder_attn: q_proj, k_proj, v_proj, out_proj)
        giúp Decoder tiếp nhận trực tiếp tín hiệu ngập ngừng (hesitation acoustics,
        F0 phẳng, năng lượng thấp) từ biểu diễn âm học của Encoder.
      + Các tầng chiếu Feed-Forward Network ("fc1", "fc2") giúp phân tách ranh giới
        âm đệm đơn ("ờ", "à", "ừm") với các âm tiết trùng âm trong từ ghép tiếng Việt.
 4. Hàm Mất mát Nhạy Từ đệm (Filler-Aware Weighted Cross-Entropy Loss):
    - Tăng trọng số phạt (filler_loss_weight=1.5) tại các vị trí token từ đệm để chống
      lại thiên kiến tự động xóa từ đệm (Verbatim Cleaning Bias) của Whisper gốc.
 5. Bộ Đo lường Kép (Dual Evaluation Metrics):
    - Tính đồng thời WER, CER và Filler Event Precision / Recall / F1 theo ranh giới từ
      (Word-boundary matching, tuyệt đối không match substring sai).
=============================================================================
"""

import os
import re
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from prepare_dut_speech_dataset import detect_filler_events_in_text, NONLEXICAL_FILLERS, DISCOURSE_FILLERS

DUT_SPEECH_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "dut_speech_16k")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "weights", "whisper_lora_vietnamese")

# Danh sách các phương ngữ miền Trung nặng được lọc bỏ khi bật chế độ chuẩn hóa giọng Miền Nam / Phổ thông
CENTRAL_HEAVY_DIALECT_KEYWORDS = ["nghe_an", "ha_tinh", "quang_binh", "quang_tri", "tp_hue"]


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], Any]]]) -> Dict[str, Any]:
        import torch

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_filler_f1_corpus(predictions: List[str], references: List[str]):
    """
    Tính Filler Event Precision, Recall và F1 theo ranh giới từ (Word-boundary matching)
    trên toàn tập đánh giá. Tuyệt đối không match substring sai ('à' trong 'bày').
    """
    tp = 0
    fp = 0
    fn = 0
    for pred, ref in zip(predictions, references):
        pred_evs, _, _ = detect_filler_events_in_text(pred)
        ref_evs, _, _ = detect_filler_events_in_text(ref)
        pred_list = [e["surface"] for e in pred_evs]
        ref_list = [e["surface"] for e in ref_evs]

        matched_ref = [False] * len(ref_list)
        for p_item in pred_list:
            found = False
            for r_i, r_item in enumerate(ref_list):
                if not matched_ref[r_i] and p_item == r_item:
                    matched_ref[r_i] = True
                    tp += 1
                    found = True
                    break
            if not found:
                fp += 1
        fn += sum(1 for m in matched_ref if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "filler_precision": round(precision * 100, 2),
        "filler_recall": round(recall * 100, 2),
        "filler_f1": round(f1 * 100, 2)
    }


def get_lora_target_modules(arch_mode: str = "extended_cross_attn") -> List[str]:
    """
    Trả về danh sách các tầng (modules) cần gắn Adapter LoRA theo chế độ kiến trúc:
    - 'baseline_qv': Chỉ can thiệp ['q_proj', 'v_proj'] (Cấu hình LoRA cơ bản).
    - 'extended_cross_attn': Can thiệp toàn diện vào cả Self-Attention lẫn Cross-Attention
      (encoder_attn: q_proj, k_proj, v_proj, out_proj) và tầng chiếu FFN (fc1, fc2).
      Nguyên lý khoa học:
        1) Trong kiến trúc Encoder-Decoder của Whisper, tầng Cross-Attention (`encoder_attn`)
           là cầu nối duy nhất truyền thông tin âm học (Log-Mel hidden states từ Encoder)
           sang bộ sinh văn bản (Decoder). Việc mở rộng LoRA vào đầy đủ `q_proj, k_proj,
           v_proj, out_proj` giúp các truy vấn (Query) của Decoder nhạy bén hơn với các đoạn
           ngập ngừng, kéo dài âm ('ờ...', 'ừm...') trên phổ âm thanh mà không bị xóa bỏ.
        2) Bổ sung LoRA vào `fc1, fc2` (Position-wise Feed-Forward Network) giúp tăng dung lượng
           biểu diễn phi tuyến để phân biệt từ đệm đứng độc lập với âm tiết trong từ ghép.
    """
    if arch_mode == "baseline_qv":
        return ["q_proj", "v_proj"]
    return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def build_filler_token_id_set(tokenizer) -> set:
    """Trích xuất tập hợp các Token ID đại diện cho từ đệm tiếng Việt để áp dụng Loss Weighting."""
    filler_token_ids = set()
    for phrase in NONLEXICAL_FILLERS + DISCOURSE_FILLERS:
        for variant in [phrase, f" {phrase}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in ids:
                filler_token_ids.add(tid)
    return filler_token_ids


def check_dependencies():
    missing_pkgs = []
    for pkg in ["torch", "transformers", "datasets", "evaluate", "peft", "soundfile", "jiwer"]:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)

    if missing_pkgs:
        print("\n" + "=" * 70)
        print("[!] THÔNG BÁO:")
        print(f"    Môi trường hiện tại đang thiếu các thư viện: {', '.join(missing_pkgs)}")
        print("    >> Cài đặt: pip install transformers datasets evaluate peft soundfile jiwer accelerate")
        print("=" * 70 + "\n")
        return False
    return True


def run_training(
    arch_mode: str = "extended_cross_attn",
    accent_filter: str = "southern_standard",
    vivos_samples: int = 250,
    filler_loss_weight: float = 1.5,
    epochs: int = 8
):
    if not check_dependencies():
        return

    import torch
    import torch.nn as nn
    import evaluate
    import pandas as pd
    from datasets import Dataset, Audio, concatenate_datasets, load_dataset
    from transformers import (
        WhisperFeatureExtractor,
        WhisperTokenizer,
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        EarlyStoppingCallback
    )
    from peft import LoraConfig, get_peft_model

    class FillerAwareSeq2SeqTrainer(Seq2SeqTrainer):
        """
        Custom Trainer cải tiến hàm mất mát (Filler-Aware Weighted Cross-Entropy Loss):
        Tăng trọng số phạt (filler_weight > 1.0) cho các vị trí token thuộc nhóm từ đệm ('ờ', 'à', 'ừm', 'kiểu như')
        nhằm khắc phục hiện tượng mất cân bằng nhãn (Class Imbalance: từ đệm chỉ chiếm <5% tổng số từ)
        và buộc Decoder không được phép tự ý lược bỏ từ đệm khi giải mã.
        """
        def __init__(self, *args, filler_token_ids=None, filler_weight=1.5, **kwargs):
            super().__init__(*args, **kwargs)
            self.filler_token_ids = filler_token_ids or set()
            self.filler_weight = float(filler_weight)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            if labels is not None and self.filler_weight > 1.0 and len(self.filler_token_ids) > 0:
                loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                per_token_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                per_token_loss = per_token_loss.view(labels.size())

                weights = torch.ones_like(per_token_loss)
                for fid in self.filler_token_ids:
                    weights = torch.where(labels == fid, self.filler_weight, weights)

                valid_mask = (labels != -100).float()
                weighted_loss = (per_token_loss * weights * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                return (weighted_loss, outputs) if return_outputs else weighted_loss

            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    print("=" * 80)
    print("   HUẤN LUYỆN PHOWHISPER + EXTENDED CROSS-ATTENTION LORA (VIVOS + SOUTHERN/STD)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị tính toán       : {device.upper()}")
    print(f"[*] Kiến trúc LoRA           : {arch_mode} -> {get_lora_target_modules(arch_mode)}")
    print(f"[*] Chế độ chuẩn hóa giọng   : {accent_filter} (Loại bỏ phương ngữ miền Trung nặng)")
    print(f"[*] Trọng số Loss Từ đệm     : gamma_filler = {filler_loss_weight}")

    MODEL_NAME = "vinai/PhoWhisper-small"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="Vietnamese", task="transcribe")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="Vietnamese", task="transcribe")

    csv_path = os.path.join(DUT_SPEECH_DIR, "metadata.csv")
    if not os.path.exists(csv_path):
        print(f"[!] Lỗi: Không tìm thấy {csv_path}. Hãy chạy prepare_dut_speech_dataset.py trước!")
        return

    df_dut = pd.read_csv(csv_path)
    df_dut["audio"] = df_dut["audio_filename"].apply(lambda fn: os.path.join(DUT_SPEECH_DIR, fn))
    df_dut = df_dut[df_dut["audio"].apply(os.path.exists)].reset_index(drop=True)

    if "split" not in df_dut.columns:
        from prepare_dut_speech_dataset import process_dataset
        process_dataset()
        df_dut = pd.read_csv(csv_path)
        df_dut["audio"] = df_dut["audio_filename"].apply(lambda fn: os.path.join(DUT_SPEECH_DIR, fn))
        df_dut = df_dut[df_dut["audio"].apply(os.path.exists)].reset_index(drop=True)

    # Chuẩn hóa tập dữ liệu theo góp ý của GVHD: Lọc bỏ phương ngữ miền Trung nặng gây underfitting
    if accent_filter == "southern_standard":
        before_cnt = len(df_dut)
        mask_central = df_dut["speaker_id"].apply(
            lambda spk: any(kw in str(spk).lower() for kw in CENTRAL_HEAVY_DIALECT_KEYWORDS)
        )
        df_filtered = df_dut[~mask_central].reset_index(drop=True)
        if len(df_filtered) >= 25:
            df_dut = df_filtered
            print(f"[+] Đã lọc bỏ {before_cnt - len(df_dut)} mẫu phương ngữ miền Trung nặng -> Giữ lại {len(df_dut)} mẫu giọng Phổ thông/Miền Nam chuẩn.")
            # Đảm bảo tập dev/test luôn có người nói độc lập sau khi lọc
            unique_spks = sorted(df_dut["canonical_speaker_id"].unique())
            if len(unique_spks) >= 3:
                test_spk = "spk_08_hong_trang" if "spk_08_hong_trang" in unique_spks else unique_spks[-1]
                rem_spks = [s for s in unique_spks if s != test_spk]
                dev_spk = rem_spks[-1]
                df_dut["split"] = df_dut["canonical_speaker_id"].apply(
                    lambda s: "test" if s == test_spk else ("dev" if s == dev_spk else "train")
                )

    df_train = df_dut[df_dut["split"] == "train"].reset_index(drop=True)
    df_dev = df_dut[df_dut["split"] == "dev"].reset_index(drop=True)
    df_test = df_dut[df_dut["split"] == "test"].reset_index(drop=True)

    print(f"[+] Chia tập theo người nói (Speaker-disjoint):")
    print(f"    • Train (Domain) : {len(df_train)} mẫu ({df_train['canonical_speaker_id'].nunique()} speakers)")
    print(f"    • Dev (Domain)   : {len(df_dev)} mẫu ({df_dev['canonical_speaker_id'].nunique()} speakers)")
    print(f"    • Test (Domain)  : {len(df_test)} mẫu ({df_test['canonical_speaker_id'].nunique()} speakers)")

    dut_train = Dataset.from_pandas(df_train[["audio", "transcription"]]).cast_column("audio", Audio(sampling_rate=16000))
    dut_dev = Dataset.from_pandas(df_dev[["audio", "transcription"]]).cast_column("audio", Audio(sampling_rate=16000))
    dut_test = Dataset.from_pandas(df_test[["audio", "transcription"]]).cast_column("audio", Audio(sampling_rate=16000))

    # Tải và tích hợp bộ ngữ liệu chuẩn VIVOS (Giọng chuẩn miền Nam / Phổ thông)
    try:
        try:
            vivos = load_dataset("vivos")
        except Exception:
            vivos = load_dataset("ademax/vivos-vie-speech2text")
        vivos = vivos.cast_column("audio", Audio(sampling_rate=16000))
        n_vivos_tr = min(vivos_samples, len(vivos["train"]))
        vivos_train = vivos["train"].select(range(n_vivos_tr))
        vivos_train = vivos_train.map(lambda b: {"transcription": b.get("sentence", b.get("transcription", ""))})
        train_ds = concatenate_datasets([dut_train, vivos_train]).shuffle(seed=42)
        print(f"    • Đã tích hợp {len(vivos_train)} mẫu VIVOS (Giọng miền Nam chuẩn) -> Tổng Train: {len(train_ds)} mẫu")
    except Exception as e:
        print(f"    [!] Chỉ dùng domain train do VIVOS chưa tải trực tuyến: {e}")
        train_ds = dut_train.shuffle(seed=42)

    eval_ds = dut_dev

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(prepare_dataset, remove_columns=eval_ds.column_names)
    test_ds = dut_test.map(prepare_dataset, remove_columns=dut_test.column_names)

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Bật hỗ trợ gradient cho input khi dùng kombinasi Gradient Checkpointing + PEFT LoRA
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # =========================================================================
    # CAN THIỆP KIẾN TRÚC LORA MỞ RỘNG (ARCHITECTURAL ENHANCEMENT)
    # Áp dụng LoRA lên cả Self-Attention, Decoder Cross-Attention (encoder_attn)
    # và tầng chiếu FFN (fc1, fc2) để bắt trọn tín hiệu ngập ngừng âm học.
    # =========================================================================
    target_modules = get_lora_target_modules(arch_mode)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
        cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)
        filler_stats = compute_filler_f1_corpus(pred_str, label_str)
        return {
            "wer": round(wer, 2),
            "cer": round(cer, 2),
            **filler_stats
        }

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    is_cuda = (device == "cuda")
    batch_size = 2 if is_cuda else 1
    grad_accum = 8 if is_cuda else 16

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=20,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=128,
        fp16=is_cuda,
        gradient_checkpointing=is_cuda,
        per_device_eval_batch_size=batch_size,
        logging_steps=15,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="filler_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none"
    )

    filler_token_ids = build_filler_token_id_set(tokenizer)
    trainer = FillerAwareSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        filler_token_ids=filler_token_ids,
        filler_weight=filler_loss_weight
    )

    print("\n[*] Bắt đầu huấn luyện tinh chỉnh Extended Cross-Attention LoRA...")
    trainer.train()

    print("\n[*] Đánh giá cuối cùng trên tập Test (Người nói độc lập chưa từng thấy khi train):")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    test_metrics["lora_architecture_mode"] = arch_mode
    test_metrics["target_modules"] = target_modules
    test_metrics["accent_filter"] = accent_filter
    test_metrics["filler_loss_weight"] = filler_loss_weight
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))

    print(f"\n[+] Lưu trọng số LoRA và cấu hình vào: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "training_eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print("\n[SUCCESS] HOÀN TẤT HUẤN LUYỆN PHOWHISPER EXTENDED LORA CHUẨN KẾ HOẠCH!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện PhoWhisper + Extended Cross-Attention LoRA")
    parser.add_argument(
        "--arch",
        type=str,
        default="extended_cross_attn",
        choices=["baseline_qv", "extended_cross_attn"],
        help="Chế độ kiến trúc LoRA: baseline_qv (chỉ q,v) hoặc extended_cross_attn (Cross-Attention + FFN)"
    )
    parser.add_argument(
        "--accent",
        type=str,
        default="southern_standard",
        choices=["southern_standard", "all_dialects"],
        help="Chế độ chuẩn hóa phương ngữ: southern_standard (bỏ giọng miền Trung nặng, chuẩn hóa VIVOS miền Nam)"
    )
    parser.add_argument("--vivos-samples", type=int, default=250, help="Số mẫu VIVOS miền Nam kết hợp huấn luyện")
    parser.add_argument("--filler-weight", type=float, default=1.5, help="Trọng số phạt hàm Loss cho token từ đệm")
    parser.add_argument("--epochs", type=int, default=8, help="Số epoch huấn luyện")
    args = parser.parse_args()

    run_training(
        arch_mode=args.arch,
        accent_filter=args.accent,
        vivos_samples=args.vivos_samples,
        filler_loss_weight=args.filler_weight,
        epochs=args.epochs
    )
