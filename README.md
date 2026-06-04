# Phishing Email Analyzer
**Nghiên cứu ATTT — FPT Corporation**

Công cụ phân tích email để phát hiện dấu hiệu tấn công phishing,
phục vụ mục đích nghiên cứu và nâng cao nhận thức an toàn thông tin.

---

## Cài đặt

```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Chạy server
python app.py

# 3. Mở trình duyệt
http://localhost:5000
```

---

## Tính năng phân tích

| Kiểm tra | Mô tả |
|---|---|
| Domain người gửi | Phát hiện typosquatting, domain giả mạo FPT |
| Giả mạo danh tính | Gmail giả danh CEO/lãnh đạo (CEO Fraud) |
| URL trong email | Domain lạ, TLD đáng ngờ, URL shortener |
| File đính kèm | Double extension (.pdf.exe), file thực thi nguy hiểm |
| Urgency keywords | Từ ngữ tạo áp lực tâm lý |
| BEC indicators | Yêu cầu chuyển tiền khẩn, giữ bí mật |
| SPF/DKIM/DMARC | Xác thực email header |
| Reply-To bất thường | From và Reply-To khác domain |

---

## Cấu trúc project

```
phishing-analyzer/
├── analyzer.py       # Logic phân tích chính
├── app.py            # Flask web server
├── requirements.txt
└── templates/
    └── index.html    # Giao diện web
```

---

## Chấm điểm nguy cơ

| Điểm | Mức độ |
|---|---|
| 0–29 | ✓ An toàn |
| 30–59 | ⚡ Đáng ngờ — Cần xác minh |
| 60–100 | ⚠ Nguy cơ cao — Phishing |

Mỗi dấu hiệu nguy hiểm cao: +25 điểm
Mỗi dấu hiệu đáng ngờ: +10 điểm
Dấu hiệu an toàn (SPF pass...): -5 điểm
