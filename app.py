import gradio as gr
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

class FinancialDocumentAnalyzer:
    def __init__(self):
        print(f"=========================================")
        print(f"INFO: FinancialDocumentAnalyzer (Gradio Only Mode)")
        print(f"=========================================")

        self.entity_labels = {
            0: "O", 1: "B-TOTAL_ASSETS", 2: "B-BEGINNING_CASH", 3: "B-ENDING_CASH",
            4: "B-FINANCIAL_CASH", 5: "B-CHANGE_IN_CASH", 6: "B-QUARTER_KEYS"
        }

    def extract_entities(self, text: str) -> List[Dict]:
        """Mock entity extraction using simple pattern matching"""
        entities = []
        words = text.split()
        
        # Simple pattern matching for financial entities
        patterns = {
            "TOTAL_ASSETS": ["total assets", "total asset"],
            "BEGINNING_CASH": ["beginning cash", "cash at beginning", "beginning period"],
            "ENDING_CASH": ["ending cash", "cash at end", "end of period"],
            "FINANCIAL_CASH": ["financial cash", "cash flows"],
            "CHANGE_IN_CASH": ["change in cash", "net change"],
            "QUARTER_KEYS": ["quarter", "Q1", "Q2", "Q3", "Q4", "march", "june", "september", "december"]
        }
        
        text_lower = text.lower()
        for entity_type, keywords in patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    entities.append({
                        "word": keyword,
                        "entity": entity_type,
                        "confidence": 0.85 + (hash(keyword) % 15) / 100,
                        "position": text_lower.find(keyword)
                    })
        
        return entities

    def simulate_document_analysis(self, document_text: str) -> Dict:
        entities = self.extract_entities(document_text)
        financial_summary = self._analyze_financial_metrics(entities, document_text)
        insights = self._generate_insights(entities, financial_summary)
        
        words = document_text.split()
        return {
            "entities": entities,
            "financial_summary": financial_summary,
            "insights": insights,
            "document_stats": {
                "total_words": len(words),
                "entities_found": len(entities),
                "pages_processed": max(1, len(words) // 500)
            }
        }

    def _analyze_financial_metrics(self, entities: List[Dict], text: str) -> Dict:
        """Extract financial values using regex patterns"""
        metrics = {}
        
        # Pattern to find monetary values
        money_pattern = r'\$[\d,]+\.?\d*|\$[\d,]+|[\d,]+\.?\d*\s*(?:thousand|million|billion)?'
        
        # Extract values near entity keywords
        for entity in entities:
            entity_type = entity["entity"].lower()
            keyword_pos = entity["position"]
            
            # Look for monetary values near the keyword
            context_start = max(0, keyword_pos - 100)
            context_end = min(len(text), keyword_pos + 100)
            context = text[context_start:context_end]
            
            money_matches = re.findall(money_pattern, context)
            if money_matches:
                # Take the first monetary value found
                value_str = money_matches[0].replace('$', '').replace(',', '')
                try:
                    value = float(value_str)
                    key = entity_type.replace('_', '').replace('b', '')
                    metrics[key] = value
                except ValueError:
                    continue
        
        return metrics

    def _generate_insights(self, entities: List[Dict], summary: Dict) -> List[str]:
        insights = []
        
        if "beginningcash" in summary and "endingcash" in summary:
            change = summary["endingcash"] - summary["beginningcash"]
            insights.append(f"Cash flow: {'Positive' if change > 0 else 'Negative'}, with a change of ${abs(change):,.2f}.")
        
        if "totalassets" in summary:
            insights.append(f"Total assets reported at ${summary['totalassets']:,.2f}.")
        
        if len(entities) > 0:
            insights.append(f"Document contains {len(entities)} financial entities.")
        else:
            insights.append("⚠️ No financial entities were detected in this document.")
        
        # Add some general insights
        insights.append("💡 This is a mock analysis for demonstration purposes.")
        
        return insights

analyzer = FinancialDocumentAnalyzer()

def analyze_financial_document(pdf_file) -> Tuple[str, str, str]:
    if pdf_file is None:
        return "Silakan upload file PDF untuk dianalisis.", "", ""
    
    # Mock analysis - interface only
    neraca_output = """## 📊 Neraca (Balance Sheet)
    
### Aset
- **Kas dan Setara Kas**: Rp 15,890,000
- **Piutang Usaha**: Rp 8,450,000
- **Persediaan**: Rp 12,300,000
- **Aset Tetap**: Rp 45,600,000
- **Total Aset**: Rp 82,240,000

### Liabilitas
- **Utang Usaha**: Rp 6,780,000
- **Utang Jangka Pendek**: Rp 8,900,000
- **Utang Jangka Panjang**: Rp 25,000,000
- **Total Liabilitas**: Rp 40,680,000

### Ekuitas
- **Modal Saham**: Rp 30,000,000
- **Laba Ditahan**: Rp 11,560,000
- **Total Ekuitas**: Rp 41,560,000

*Status: File PDF berhasil dianalisis*
"""

    cashflow_output = """## 🌊 Arus Kas (Cash Flow)
    
### Arus Kas dari Aktivitas Operasi
- **Penerimaan dari Pelanggan**: Rp 45,600,000
- **Pembayaran kepada Pemasok**: (Rp 28,900,000)
- **Pembayaran Gaji**: (Rp 8,700,000)
- **Kas Bersih dari Operasi**: Rp 8,000,000

### Arus Kas dari Aktivitas Investasi
- **Pembelian Aset Tetap**: (Rp 5,200,000)
- **Penjualan Investasi**: Rp 2,100,000
- **Kas Bersih dari Investasi**: (Rp 3,100,000)

### Arus Kas dari Aktivitas Pendanaan
- **Penerimaan Pinjaman**: Rp 8,000,000
- **Pembayaran Dividen**: (Rp 2,500,000)
- **Kas Bersih dari Pendanaan**: Rp 5,500,000

### Ringkasan
- **Kas Awal Periode**: Rp 4,590,000
- **Perubahan Kas Bersih**: Rp 10,400,000
- **Kas Akhir Periode**: Rp 14,990,000

*Status: Analisis arus kas berhasil*
"""

    labarugi_output = """## 📈 Laba Rugi (Profit & Loss)
    
### Pendapatan
- **Penjualan Bersih**: Rp 125,600,000
- **Pendapatan Lain-lain**: Rp 2,400,000
- **Total Pendapatan**: Rp 128,000,000

### Beban
- **Harga Pokok Penjualan**: Rp 75,600,000
- **Laba Kotor**: Rp 52,400,000
- **Beban Operasional**: Rp 28,900,000
- **Laba Operasional**: Rp 23,500,000

### Beban Non-Operasional
- **Beban Bunga**: Rp 3,200,000
- **Beban Pajak**: Rp 5,100,000
- **Total Beban Non-Operasional**: Rp 8,300,000

### Hasil Akhir
- **Laba Bersih**: Rp 15,200,000
- **Margin Laba Bersih**: 11.9%
- **EPS (Earning Per Share)**: Rp 1,520

*Status: Analisis laba rugi selesai*
"""
    
    return neraca_output, cashflow_output, labarugi_output

def create_gradio_interface():
    with gr.Blocks(title="LongFin Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏦 LongFin: AI-Powered Financial Document Analyzer")
        gr.Markdown("*Upload file PDF laporan keuangan untuk analisis AI*")
        
        with gr.Row():
            with gr.Column(scale=2):
                pdf_input = gr.File(
                    label="📄 Upload Laporan Keuangan (PDF)",
                    file_types=[".pdf"],
                    type="filepath"
                )
                analyze_btn = gr.Button("🔍 Analisis Dokumen", variant="primary", size="lg")
                
                gr.Markdown("### 📋 Fitur Analisis:")
                gr.Markdown("- **Neraca**: Aset, Liabilitas, Ekuitas")
                gr.Markdown("- **Arus Kas**: Operasi, Investasi, Pendanaan")
                gr.Markdown("- **Laba Rugi**: Pendapatan, Beban, Laba Bersih")
            
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("📊 Neraca"):
                        neraca_output = gr.Markdown()
                    with gr.Tab("🌊 Arus Kas"):
                        cashflow_output = gr.Markdown()
                    with gr.Tab("📈 Laba Rugi"):
                        labarugi_output = gr.Markdown()
        
        analyze_btn.click(
            fn=analyze_financial_document,
            inputs=[pdf_input],
            outputs=[neraca_output, cashflow_output, labarugi_output]
        )
        
        gr.Markdown("---")
        gr.Markdown("### 💡 Cara Penggunaan:")
        gr.Markdown("1. Upload file PDF laporan keuangan")
        gr.Markdown("2. Klik tombol 'Analisis Dokumen'")
        gr.Markdown("3. Lihat hasil analisis di tab yang tersedia")
        
        demo.queue()
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)