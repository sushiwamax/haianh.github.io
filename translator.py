from transformers import pipeline

def translate_text(text, source_lang, target_lang):
    """
    Dịch văn bản từ source_lang sang target_lang.
    Sử dụng mô hình NLLB từ Meta (facebook/nllb-200-distilled-600M).
    Hỗ trợ hơn 200 ngôn ngữ.
    """
    try:
        # Ánh xạ mã ngôn ngữ ngắn sang mã đầy đủ của NLLB
        # (NLLB yêu cầu định dạng như 'eng_Latn' thay vì 'en')
        lang_map = {
            'en': 'eng_Latn',  # Tiếng Anh
            'vi': 'vie_Latn',  # Tiếng Việt
            'fr': 'fra_Latn',  # Tiếng Pháp
            'es': 'spa_Latn',  # Tiếng Tây Ban Nha
            'de': 'deu_Latn',  # Tiếng Đức
            'zh': 'zho_Hans',  # Tiếng Trung (giản thể)
            'ja': 'jpn_Jpan',  # Tiếng Nhật
            'ko': 'kor_Hang',  # Tiếng Hàn
            'ar': 'ara_Arab',  # Tiếng Ả Rập
            'ru': 'rus_Cyrl',  # Tiếng Nga
            # Thêm ngôn ngữ khác nếu cần (xem danh sách tại https://huggingface.co/facebook/nllb-200-distilled-600M)
        }
        
        # Lấy mã ngôn ngữ đầy đủ, fallback nếu không có
        src_lang_full = lang_map.get(source_lang, f"{source_lang}_Latn")
        tgt_lang_full = lang_map.get(target_lang, f"{target_lang}_Latn")
        
        # Kiểm tra nếu ngôn ngữ không được hỗ trợ
        if src_lang_full == f"{source_lang}_Latn" or tgt_lang_full == f"{target_lang}_Latn":
            raise ValueError(f"Ngôn ngữ '{source_lang}' hoặc '{target_lang}' không được hỗ trợ. Kiểm tra ánh xạ.")
        
        # Tải pipeline với mô hình NLLB distilled (nhỏ, nhanh)
        translator = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang=src_lang_full,
            tgt_lang=tgt_lang_full,
            device=-1  # Buộc dùng CPU nếu GPU lỗi (thay -1 bằng 0 nếu có GPU)
        )
        
        # Dịch văn bản (giới hạn độ dài để tránh lỗi)
        result = translator(text, max_length=512, truncation=True)
        return result[0]['translation_text']
    
    except ValueError as e:
        raise Exception(f"Lỗi ngôn ngữ: {str(e)}")
    except Exception as e:
        raise Exception(f"Dịch thất bại: {str(e)}. Kiểm tra cài đặt transformers, RAM, hoặc kết nối internet (cho lần tải đầu).")
