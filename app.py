import gradio as gr
import re
import torch
import PyPDF2
# Ganti atau pastikan impor ini ada di bagian atas file
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings('ignore')

class DocumentAnalyzer:
    """
    Document Analyzer using a T5 model for summarization.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.model_loaded = False

        # Model yang akan digunakan dari Hugging Face Hub
        model_name = "jiryanfarokhi997/Qwen-Financial-Analysis-v1"

        print(f"Loading model: {model_name}...")

        try:
            # 1. Muat tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 2. Muat model Seq2Seq
            # Ubah menjadi seperti ini
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

            self.model_loaded = True
            print(f"Model '{model_name}' loaded successfully on {self.device}")

        except Exception as e:
            print(f"FATAL: Error loading model: {e}")
            print("Model could not be loaded. Will use rule-based analysis instead.")
            self.model_loaded = False

    def extract_text_from_pdf(self, pdf_file):
        if pdf_file is None:
            return ""
        try:
            with open(pdf_file.name, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    # Ganti fungsi preprocess_text yang lama dengan yang ini

    def preprocess_text(self, text):
        """
        Cleans up text extracted from a PDF to improve its quality for the AI model.
        """
        # 1. Menangani kata yang terpotong oleh hyphen di akhir baris
        # Contoh: "dokumen-\ntasi" -> "dokumentasi"
        text = re.sub(r'-\n', '', text)

        # 2. Menggabungkan kalimat yang terpotong oleh baris baru.
        # Baris baru akan diganti spasi, KECUALI jika baris sebelumnya diakhiri titik.
        # Ini membantu menyatukan paragraf yang terpecah.
        text = re.sub(r'(?<!\.)\n', ' ', text)
        # Menjaga baris baru yang memang menandakan akhir paragraf
        text = re.sub(r'\n', '\n\n', text)

        # 3. Menghapus baris yang kemungkinan besar adalah nomor halaman atau header/footer
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            # Hapus baris jika hanya berisi angka, spasi, atau kata "Page" diikuti angka.
            if not re.fullmatch(r'\s*(\d+|Page\s+\d+)\s*', line.strip(), re.IGNORECASE):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)

        # 4. Menghapus spasi ganda dan spasi di awal/akhir
        text = re.sub(r'\s+', ' ', text).strip()
        
        print("--- Preprocessed Text Sample ---")
        print(text[:500]) # Mencetak 500 karakter pertama dari teks yang sudah bersih untuk pengecekan
        print("------------------------------")

        return text
    def analyze_document(self, document_text):
        if not document_text.strip():
            return "Tidak ada teks yang dapat dianalisis dari dokumen."

        if not self.model_loaded:
            print("Model not loaded, using rule-based analysis")
            return self.create_fallback_analysis(document_text)

        processed_text = self.preprocess_text(document_text)

        # T5 bekerja dengan baik dengan prefix instruksi
        prompt = f"summarize: {processed_text}"
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=1024, # Batas token input
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=256,  # Batas maksimal token untuk ringkasan
                    num_beams=4,      # Menggunakan beam search untuk hasil yang lebih baik
                    early_stopping=True
                )
            
            # Output dari model T5 adalah ringkasan itu sendiri
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if not generated_text:
                return self.create_fallback_analysis(document_text)
            
            return self.format_analysis_output(generated_text, document_text)
            
        except Exception as e:
            print(f"Error in model inference: {e}")
            return self.create_fallback_analysis(document_text)

    def create_fallback_analysis(self, document_text):
        print("Using rule-based analysis fallback")
        return "## ⚠️ Peringatan\nModel AI gagal dimuat atau gagal menghasilkan output. Pastikan koneksi internet stabil dan coba lagi."

    def format_analysis_output(self, generated_text, original_text):
        word_count = len(original_text.split())
        
        formatted_output = f"""
# 📄 Ringkasan Dokumen

## ✨ Ringkasan oleh AI

{generated_text}

---

## 📋 Informasi Dokumen
- **Total Kata**: {word_count}
- **Model**: mrSoul7766/simpleT5-Base-ECTSum
- **Status**: Ringkasan selesai
"""
        return formatted_output

# Inisialisasi analyzer di luar fungsi agar hanya dimuat sekali
analyzer = DocumentAnalyzer()

def process_document(pdf_file):
    if pdf_file is None:
        return "❌ Silakan unggah file PDF untuk diringkas."
    
    document_text = analyzer.extract_text_from_pdf(pdf_file)
    if not document_text.strip():
        return "❌ Gagal mengekstrak teks dari PDF. Pastikan file berisi teks dan tidak terproteksi."
    
    return analyzer.analyze_document(document_text)

def create_gradio_interface():
    with gr.Blocks(title="Document Summarizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📄 AI Document Summarizer")
        gr.Markdown("*Didukung oleh model T5 dari Hugging Face*")
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="📄 Unggah Dokumen (PDF)")
                analyze_btn = gr.Button("✨ Buat Ringkasan", variant="primary")
                gr.Markdown("---")
                gr.Markdown("### 🤖 Info Model")
                gr.Markdown("- **Model**: `mrSoul7766/simpleT5-Base-ECTSum`\n- **Tugas**: Meringkas Teks")

            with gr.Column(scale=2):
                output_display = gr.Markdown(label="Hasil Ringkasan", value="Unggah dokumen PDF dan klik 'Buat Ringkasan' untuk memulai.")
        
        analyze_btn.click(
            fn=process_document,
            inputs=[pdf_input],
            outputs=[output_display],
            api_name="summarize_pdf"
        )
        
        demo.queue()
    return demo

if __name__ == "__main__":
    app_interface = create_gradio_interface()
    app_interface.launch(server_name="0.0.0.0", server_port=7860)
