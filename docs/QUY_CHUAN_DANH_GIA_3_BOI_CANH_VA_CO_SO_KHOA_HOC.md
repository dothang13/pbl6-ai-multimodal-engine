# TÀI LIỆU QUY CHUẨN ĐÁNH GIÁ 3 BỐI CẢNH & CƠ SỞ KHOA HỌC HỆ THỐNG
## (Specification for 3 Presentation Contexts, Rubrics & Scientific References)
> **Dự án:** Nền tảng Đánh giá Kỹ năng Thuyết trình Tự động Đa Phương thức (PBL6 - DUT)  
> **Mục đích:** Tài liệu kỹ thuật chuẩn xác để thiết kế Database Schema (PostgreSQL 16), Validate Input/Output API Gateway (FastAPI) và điều phối Worker (Celery/Redis).

---

## 1. TỔNG QUAN 3 BỐI CẢNH THUYẾT TRÌNH (PRESENTATION CONTEXTS)

Hệ thống AI không chấm điểm theo một công thức cứng nhắc mà **thích ứng thông minh theo 3 bối cảnh thực tế** thông qua tham số `context_type`:

| Mã Bối Cảnh (`context_type`) | Tên Tiếng Việt | Đối tượng / Trường hợp sử dụng | Đặc trưng cốt lõi cần đánh giá |
| :--- | :--- | :--- | :--- |
| **`do_an`** | **Thuyết trình Đồ án & Học thuật** *(Academic & Capstone Defense)* | Sinh viên bảo vệ đồ án môn học, tốt nghiệp, hội thảo khoa học trước Hội đồng Giảng viên. | Đề cao **tư thế đĩnh đạc, ánh mắt tập trung vào hội đồng, nhịp nói kỹ thuật chuẩn xác (115–130 WPM), triệt tiêu tối đa từ đệm ậm ờ** (*ờ, à, ừm*). Biểu cảm giữ sự điềm đạm, nghiêm túc vừa phải. |
| **`hung_bien`** | **Hùng biện & Tranh biện Công chúng** *(Public Speaking & Debate)* | Diễn giả, MC, sinh viên tham gia tranh biện, truyền cảm hứng trước đám đông. | Đề cao **ngôn ngữ cơ thể linh hoạt, nụ cười tự nhiên, biên độ biểu cảm cơ mặt sinh động và độ biến thiên ngữ điệu giọng nói ($F_0$ Pitch SD)** để tạo cảm xúc dẫn dắt khán giả. |
| **`pitching`** | **Pitching Gọi vốn Khởi nghiệp** *(Startup & Investor Pitching)* | Sinh viên, founder thuyết trình dự án khởi nghiệp trước các nhà đầu tư (Shark Tank, Demo Day). | Đề cao **năng lượng tích cực, nụ cười tự tin chuẩn Action Unit AU12, tương tác mắt thuyết phục và tốc độ nói dứt khoát quyết đoán (135–155 WPM)** nhằm tạo độ tin cậy tài chính. |

---

## 2. MA TRẬN TRỌNG SỐ & BAREM ĐIỂM CHUẨN (RUBRIC WEIGHT MATRIX)

Nhân lưu ý: Điểm tổng kết (`overall_score`) được tính từ **5 tiêu chí thành phần** (thang điểm 100), trong đó mỗi bối cảnh có **bộ trọng số riêng biệt** (tổng bằng 1.00):

$$\text{Overall Score} = w_1 \cdot S_{\text{eye}} + w_2 \cdot S_{\text{posture}} + w_3 \cdot S_{\text{facial}} + w_4 \cdot S_{\text{speech}} + w_5 \cdot S_{\text{filler}}$$

### Bảng Ma trận Trọng số theo Bối cảnh:
| Tiêu chí Rubric (Trường JSON) | Bối cảnh `do_an` | Bối cảnh `hung_bien` | Bối cảnh `pitching` | Ý nghĩa hành vi & Chỉ số cảm biến |
| :--- | :---: | :---: | :---: | :--- |
| **`eye_contact_score`** ($S_{\text{eye}}$) | **25%** ($0.25$) | **20%** ($0.20$) | **25%** ($0.25$) | Tỷ lệ nhìn thẳng camera $\ge 75\%$ thời lượng; góc lệch mắt $|Yaw| < 15^\circ, |Pitch| < 15^\circ$. |
| **`head_posture_score`** ($S_{\text{posture}}$) | **20%** ($0.20$) | **15%** ($0.15$) | **15%** ($0.15$) | Độ ổn định đầu, tỷ lệ thời gian không bị lắc/cúi/nghiêng $>15^\circ$ (lọc qua 1-Euro Filter). |
| **`facial_expression_score`** ($S_{\text{facial}}$) | **10%** ($0.10$) | **20%** ($0.20$) | **25%** ($0.25$) | Nụ cười Action Unit AU12 (*mouthSmile*) & Năng lượng cơ mặt (Pitching cần cười tự tin $\ge 30\%$). |
| **`speech_rate_score`** ($S_{\text{speech}}$) | **25%** ($0.25$) | **25%** ($0.25$) | **20%** ($0.20$) | Tốc độ nói WPM và độ biến thiên ngữ điệu giọng ($F_0$ Pitch SD $\ge 25\text{Hz}$ tránh đều đều gây buồn ngủ). |
| **`filler_control_score`** ($S_{\text{filler}}$) | **20%** ($0.20$) | **20%** ($0.20$) | **15%** ($0.15$) | Kiểm soát từ đệm (*ờ, à, ừm, kiểu như*): 0 từ/phút = 100đ, $\le 2$ từ/phút = 85đ, $\ge 5$ từ/phút $< 50$đ. |
| **TỔNG HỢP (`overall_score`)** | **100%** | **100%** | **100%** | Điểm số tổng quát của bài thuyết trình (Thang điểm 0.0 – 100.0). |

---

## 3. CƠ SỞ KHOA HỌC TỪ CÁC BÀI BÁO QUỐC TẾ (ACADEMIC REFERENCES)

Để Nhân tự tin khi viết tài liệu kiến trúc Backend hoặc trả lời câu hỏi của thầy Trí về căn cứ xây dựng barem điểm, đây là các bài báo khoa học chuẩn mực quốc tế mà nhóm AI đã kế thừa và phát triển:

### 3.1. Về Thị giác & Cử chỉ Không lời (Computer Vision & Non-verbal Cues)
1. **Ekman & Friesen (FACS - Facial Action Coding System):**  
   - *Cơ sở áp dụng:* Tiêu chuẩn quốc tế để đo nụ cười qua **Action Unit AU12** (*Lip Corner Puller*) bằng 52 blendshapes của MediaPipe. Phân biệt nụ cười tự nhiên với cười gượng gạo trong bối cảnh Pitching.
2. **Chen et al. (ICMI) & Ristorcelli et al. (ACM IVA 2025):**  
   - *Cơ sở áp dụng:* Chứng minh ngưỡng giao tiếp ánh mắt hiệu quả trong thuyết trình công chúng là tỷ lệ nhìn vào người nghe đạt từ **$70\% - 80\%$**, và góc lệch đầu vượt quá **$15^\circ$** được coi là hành vi phân tâm (Looking Away).
3. **Casiez et al. (CHI) – Thuật toán `1-Euro Filter`:**  
   - *Cơ sở áp dụng:* Bộ lọc thích ứng khử triệt để hiện tượng rung giật (jitter) của webcam khi tính toán Yaw/Pitch/Roll, giảm sai số đo góc xuống mức lý tưởng $0.13^\circ$.

### 3.2. Về Âm học & Nhận dạng Tiếng nói Giữ nguyên Từ đệm (Acoustic & Verbatim ASR)
4. **Le et al. (VinAI Research – ICLR 2024) – *PhoWhisper: ASR for Vietnamese*:**  
   - *Cơ sở áp dụng:* Mô hình nền tảng SOTA tiếng Việt với 844 giờ dữ liệu chuẩn. Đề tài kế thừa làm Base Model và tinh chỉnh LoRA mở rộng (Cross-Attention) để không bị "quên" ngữ âm tiếng Việt.
5. **Wagner et al. (Interspeech 2024) – *CrisperWhisper* & Smith et al. (Apple – Interspeech 2025):**  
   - *Cơ sở áp dụng:* Các công trình mới nhất khẳng định ASR thương mại (như Whisper gốc) luôn tự động xóa từ đệm (*Intended-speech Bias*). Đề tài thiết kế **Filler-Aware Loss ($\gamma=1.5$)** để ép mô hình bắt trọn vẹn từ đệm tiếng Việt (*ờ, à, ừm, kiểu như*).
6. **Toastmasters International & ABET Communication Standard:**  
   - *Cơ sở áp dụng:* Tốc độ nói chuẩn của bài thuyết trình kỹ thuật là **115–130 WPM**, bài hùng biện/gọi vốn là **130–155 WPM**. Vượt quá 165 WPM là nói quá nhanh; dưới 100 WPM là ấp úng.

### 3.3. Về Học máy Chấm điểm & Giải thích Hộp trắng (Tabular ML & Explainable AI)
7. **Prokhorenkova et al. (NeurIPS) – *CatBoost: Unbiased boosting with categorical features*:**  
   - *Cơ sở áp dụng:* Cấu trúc cây đối xứng (Oblivious Trees) của CatBoost kháng hiện tượng Overfitting trên vector 123 chiều vượt trội so với XGBoost/LightGBM.
8. **Lundberg & Lee (NeurIPS) – *A Unified Approach to Interpreting Model Predictions (SHAP)*:**  
   - *Cơ sở áp dụng:* Dùng **SHAP TreeExplainer** tính toán giá trị đóng góp biên Shapley $\phi_i$ cho từng đặc trưng, giúp Backend có ngay dữ liệu "Top 5 điểm trừ mạnh nhất" và "Top 5 điểm cộng" để hiển thị cho sinh viên mà không cần đoán mò.

---

## 4. CONTRACT DỮ LIỆU JSON ĐỂ NHÂN TÍCH HỢP VÀO POSTGRESQL & API

Khi Celery Worker gọi `ai_assessment_engine.py`, kết quả trả về sẽ có cấu trúc JSON chuẩn dưới đây. Nhân hãy map vào các cột Database (PostgreSQL 16) tương ứng:

```json
{
  "video_id": "DA_001",
  "context_type": "do_an",
  "duration_sec": 182.4,
  
  // 1. Lưu vào cột: overall_score (DOUBLE PRECISION)
  "overall_score": 84.65,

  // 2. Lưu vào cột: rubric_scores (JSONB)
  "rubric_scores": {
    "eye_contact_score": 86.5,
    "head_posture_score": 88.0,
    "facial_expression_score": 75.0,
    "speech_rate_score": 85.2,
    "filler_control_score": 88.5
  },

  // 3. Lưu vào cột: xai_explanation (JSONB) - Phục vụ hiển thị Dashboard
  "shap_explanation": {
    "base_score": 78.5,
    "top_penalties": [
      {
        "feature": "aud_05_filler_words_per_min",
        "feature_vi": "Tần suất phát âm từ đệm (ờ, à, ừm) mỗi phút",
        "value": 3.8,
        "shap_impact": -4.25
      },
      {
        "feature": "vis_17_posture_violation_ratio",
        "feature_vi": "Tỷ lệ thời gian cúi/nghiêng đầu vượt ngưỡng 15°",
        "value": 0.22,
        "shap_impact": -2.80
      }
    ],
    "top_strengths": [
      {
        "feature": "vis_02_eye_contact_ratio",
        "feature_vi": "Tỷ lệ duy trì giao tiếp ánh mắt với camera",
        "value": 0.82,
        "shap_impact": 5.10
      }
    ]
  },

  // 4. Lưu vào cột: telemetry_timeseries (JSONB) - Phục vụ thanh tua Seekbar cho Sơn (Frontend)
  "telemetry_timeseries": [
    {
      "timestamp_sec": 12.0,
      "start_ms": 12000,
      "eye_contact": false,
      "head_pose": { "yaw": 18.2, "pitch": -6.5, "roll": 2.1 },
      "wpm_instant": 140,
      "is_violation": true,
      "violation_type": "Mất tương tác ánh mắt với camera; Lệch tư thế đầu (Yaw=18.2°)"
    }
  ]
}
```

---
*Tài liệu kỹ thuật chuẩn phục vụ phát triển hệ thống và triển khai Backend/DevOps PBL6.*

