from flask import Flask, request, render_template
from translator import translate_text  # Import hàm từ translator.py

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    translated = ""
    error = ""
    if request.method == 'POST':
        text = request.form.get('text')
        source_lang = request.form.get('source_lang')
        target_lang = request.form.get('target_lang')
        if text:
            try:
                translated = translate_text(text, source_lang, target_lang)
            except Exception as e:
                error = f"Lỗi dịch: {str(e)}"  # Hiển thị lỗi rõ ràng
        else:
            error = "Vui lòng nhập văn bản cần dịch."
    return render_template("index.html", translated=translated, error=error)

if __name__ == '__main__':
    app.run(debug=True)