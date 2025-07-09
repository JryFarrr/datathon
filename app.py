import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import json
import re
import os
import psutil
import PyPDF2
import fitz  # PyMuPDF
import tempfile
from io import BytesIO

class FinancialAnalyzer:
    def __init__(self):
        # Check system resources first
        self.check_system_resources()
        
        # Model options ordered by resource requirements (lightest first)
        self.model_options = [
            {
                "name": "google/flan-t5-small",
                "type": "lightweight",
                "memory_req": "~1GB",
                "description": "Very lightweight instruction-following model"
            },
            {
                "name": "google/flan-t5-base", 
                "type": "lightweight",
                "memory_req": "~2GB",
                "description": "Good balance of performance and efficiency"
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "type": "lightweight", 
                "memory_req": "~1.5GB",
                "description": "Conversational model for financial analysis"
            },
            {
                "name": "microsoft/Phi-4-reasoning-plus",
                "type": "heavy",
                "memory_req": "~8GB+",
                "description": "Advanced reasoning model (requires substantial resources)",
                "lora_adapter": "kingabzpro/Phi-4-Reasoning-Plus-FinQA-COT"
            }
        ]
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.current_model = None
        self.load_model()
    
    def check_system_resources(self):
        """Check available system resources"""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        gpu_available = torch.cuda.is_available()
        
        print(f"System RAM: {ram_gb:.1f} GB")
        print(f"GPU Available: {gpu_available}")
        
        if gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            
            # Recommend model based on available resources
            if gpu_memory < 4:
                print("⚠️  GPU memory < 4GB - Recommend lightweight models only")
                self.recommended_models = ["lightweight"]
            elif gpu_memory < 8:
                print("⚠️  GPU memory < 8GB - Phi-4 may struggle")
                self.recommended_models = ["lightweight"]
            else:
                print("✅ GPU memory sufficient for heavier models")
                self.recommended_models = ["lightweight", "heavy"]
        else:
            print("⚠️  No GPU detected - Using CPU (slower performance)")
            self.recommended_models = ["lightweight"]
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file using multiple methods - FIXED VERSION"""
        text = ""
        
        try:
            # Check if pdf_file is None
            if pdf_file is None:
                return "❌ Error: No PDF file provided."
            
            # Handle different input types from Gradio
            if hasattr(pdf_file, 'name'):
                # pdf_file is a file object with .name attribute
                pdf_path = pdf_file.name
                print(f"📄 Processing PDF: {pdf_path}")
            elif isinstance(pdf_file, str):
                # pdf_file is a file path string
                pdf_path = pdf_file
                print(f"📄 Processing PDF: {pdf_path}")
            else:
                # pdf_file might be file content
                return "❌ Error: Unsupported PDF file format."
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                return "❌ Error: PDF file not found."
            
            # Method 1: Try PyMuPDF first (better for complex layouts)
            try:
                print("🔄 Trying PyMuPDF extraction...")
                doc = fitz.open(pdf_path)
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    text += page_text
                
                doc.close()
                print(f"✅ PyMuPDF extraction successful ({len(text)} characters)")
                
            except Exception as e:
                print(f"❌ PyMuPDF failed: {e}")
                
                # Method 2: Fallback to PyPDF2
                try:
                    print("🔄 Trying PyPDF2 extraction...")
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            text += page_text
                    
                    print(f"✅ PyPDF2 extraction successful ({len(text)} characters)")
                    
                except Exception as e2:
                    print(f"❌ PyPDF2 also failed: {e2}")
                    return f"❌ Error: Could not extract text from PDF. PyMuPDF error: {e}, PyPDF2 error: {e2}"
            
            # Clean up extracted text
            if text:
                # Remove excessive whitespace and clean up
                text = re.sub(r'\n+', '\n', text)
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                
                # If text is too short, it might be a scan or protected PDF
                if len(text) < 50:
                    return "⚠️ Warning: Very little text extracted. This might be a scanned PDF or protected document. Please try an OCR tool or convert to text manually."
                
                print(f"✅ Text cleaning completed. Final length: {len(text)} characters")
                return text
            else:
                return "❌ Error: No text could be extracted from the PDF."
                
        except Exception as e:
            print(f"❌ General PDF processing error: {e}")
            return f"❌ Error processing PDF: {str(e)}"
    
    def load_phi4_model(self, model_config):
        """Load Phi-4 model with LoRA adapter"""
        try:
            print(f"Loading Phi-4 base model: {model_config['name']}")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config['name'],
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load LoRA adapter
            print(f"Loading LoRA adapter: {model_config['lora_adapter']}")
            self.model = PeftModel.from_pretrained(
                base_model,
                model_config['lora_adapter'],
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config['name'], 
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.current_model = model_config['name']
            print(f"✅ Successfully loaded Phi-4 with LoRA adapter")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Phi-4: {str(e)}")
            return False
    
    def load_lightweight_model(self, model_name):
        """Load lightweight model"""
        try:
            print(f"Loading lightweight model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.current_model = model_name
            print(f"✅ Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)}")
            return False
    
    def load_model(self):
        """Load best available model based on system resources"""
        
        # Try models in order of preference
        for model_config in self.model_options:
            # Skip heavy models if not recommended
            if model_config['type'] not in self.recommended_models:
                print(f"⏭️  Skipping {model_config['name']} - insufficient resources")
                continue
            
            print(f"🔄 Trying {model_config['name']} ({model_config['memory_req']})")
            
            if model_config['type'] == 'heavy':
                if self.load_phi4_model(model_config):
                    break
            else:
                if self.load_lightweight_model(model_config['name']):
                    break
        
        # Fallback to rule-based if all models fail
        if self.model is None:
            print("⚠️  All model loading attempts failed. Using rule-based analysis.")
            self.current_model = "rule-based"
    
    def generate_response_phi4(self, prompt, max_length=1200):
        """Generate response using Phi-4 model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove original prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response if response else "No response generated."
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_response_lightweight(self, prompt, max_length=512):
        """Generate response using lightweight model"""
        try:
            response = self.pipeline(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                truncation=True
            )
            
            generated_text = response[0]['generated_text']
            
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            return generated_text if generated_text else "No response generated."
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_response(self, prompt, max_length=512):
        """Generate response using available model"""
        if self.current_model == "rule-based":
            return "Unable to generate AI response. Please check the rule-based analysis."
        
        # Use appropriate generation method based on model type
        if "Phi-4" in self.current_model:
            return self.generate_response_phi4(prompt, max_length)
        else:
            return self.generate_response_lightweight(prompt, max_length)
    
    def analyze_financial_report_rule_based(self, financial_text):
        """Rule-based financial analysis as fallback"""
        if financial_text.startswith("❌") or financial_text.startswith("⚠️"):
            return financial_text
        
        text_lower = financial_text.lower()
        
        # Extract key financial metrics
        metrics = {
            'revenue': ['revenue', 'sales', 'total revenue', 'net sales'],
            'profit': ['profit', 'net income', 'earnings', 'net profit'],
            'cash': ['cash', 'cash flow', 'operating cash flow'],
            'assets': ['total assets', 'assets', 'current assets'],
            'liabilities': ['liabilities', 'debt', 'total liabilities'],
            'equity': ['equity', 'shareholders equity', 'stockholders equity']
        }
        
        found_metrics = {}
        for category, keywords in metrics.items():
            for keyword in keywords:
                if keyword in text_lower:
                    pattern = rf'{keyword}[:\s]*[\$]?([0-9,]+\.?[0-9]*[MmBbKk]?)'
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        found_metrics[category] = matches[0]
        
        # Generate analysis
        analysis_text = f"""
# 📊 Financial Analysis Report
*Generated using: {self.current_model}*
*Source: PDF Document ({len(financial_text)} characters extracted)*

## 1. Financial Statement Analysis
"""
        
        if 'revenue' in found_metrics:
            analysis_text += f"- **Revenue**: ${found_metrics['revenue']}\n"
        if 'profit' in found_metrics:
            analysis_text += f"- **Net Profit**: ${found_metrics['profit']}\n"
        
        analysis_text += """
### Key Observations:
- Revenue trends indicate the company's ability to generate sales
- Profit margins show operational efficiency
- Year-over-year growth patterns reveal business trajectory

## 2. Cash Flow Analysis
"""
        
        if 'cash' in found_metrics:
            analysis_text += f"- **Cash Flow**: ${found_metrics['cash']}\n"
        
        analysis_text += """
### Key Observations:
- Operating cash flow shows core business performance
- Free cash flow indicates available funds for growth
- Cash management reflects financial discipline

## 3. Balance Sheet Analysis
"""
        
        if 'assets' in found_metrics:
            analysis_text += f"- **Total Assets**: ${found_metrics['assets']}\n"
        if 'liabilities' in found_metrics:
            analysis_text += f"- **Total Liabilities**: ${found_metrics['liabilities']}\n"
        if 'equity' in found_metrics:
            analysis_text += f"- **Shareholders' Equity**: ${found_metrics['equity']}\n"
        
        analysis_text += """
### Key Observations:
- Asset composition shows investment allocation
- Debt levels indicate financial leverage
- Equity position reflects shareholder value

## Summary
This analysis provides a comprehensive view of the company's financial health across three critical dimensions.
"""
        
        return analysis_text
    
    def analyze_financial_report(self, financial_text):
        """Analyze financial report focusing on three main aspects"""
        
        # Check if text extraction failed
        if financial_text.startswith("❌") or financial_text.startswith("⚠️"):
            return financial_text
        
        # First try rule-based analysis
        rule_based_result = self.analyze_financial_report_rule_based(financial_text)
        
        if self.current_model == "rule-based":
            return rule_based_result
        
        # Create specialized prompt for financial analysis
        if "Phi-4" in self.current_model:
            analysis_prompt = f"""<|user|>
You are a financial analyst. Analyze this financial report and provide detailed insights focusing on:

1. **Financial Statement Analysis**: Revenue trends, profitability, expense management
2. **Cash Flow Analysis**: Operating cash flow, free cash flow, cash management
3. **Balance Sheet Analysis**: Asset composition, debt levels, equity position

Financial Report:
{financial_text[:2000]}

Provide a comprehensive analysis with specific insights and recommendations.
<|assistant|>"""
            max_tokens = 1000
        else:
            analysis_prompt = f"""Analyze this financial report focusing on:

1. Financial Statement Analysis (revenue, profitability, expenses)
2. Cash Flow Analysis (operating, investing, financing)  
3. Balance Sheet Analysis (assets, liabilities, ratios)

Financial Report:
{financial_text[:1000]}...

Provide detailed insights for each aspect:"""
            max_tokens = 600
        
        ai_result = self.generate_response(analysis_prompt, max_tokens)
        
        return f"{rule_based_result}\n\n---\n\n## 🤖 AI Analysis:\n{ai_result}"
    
    def answer_question(self, financial_text, question):
        """Answer specific questions about the financial report"""
        
        # Check if text extraction failed
        if financial_text.startswith("❌") or financial_text.startswith("⚠️"):
            return financial_text
        
        if not question.strip():
            return "Please provide a question about the financial report."
        
        if self.current_model == "rule-based":
            return f"Question: {question}\n\nAnswer: Unable to provide AI-generated answers. Please refer to the main analysis above for general insights."
        
        if "Phi-4" in self.current_model:
            question_prompt = f"""<|user|>
Based on this financial report, answer the following question with detailed analysis:

Question: {question}

Financial Report:
{financial_text[:1500]}

Provide a comprehensive answer with supporting evidence from the report.
<|assistant|>"""
            max_tokens = 600
        else:
            question_prompt = f"""Based on this financial report, answer the question:

Question: {question}

Financial Report:
{financial_text[:800]}...

Answer:"""
            max_tokens = 400
        
        return self.generate_response(question_prompt, max_tokens)
    
    def get_model_info(self):
        """Get current model information"""
        model_info = f"**Current Model**: {self.current_model}\n"
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            model_info += f"**GPU Memory**: {gpu_memory:.1f} GB\n"
        
        model_info += f"**System RAM**: {psutil.virtual_memory().total / (1024**3):.1f} GB"
        
        return model_info

# Initialize analyzer
print("🚀 Initializing Financial Analyzer...")
analyzer = FinancialAnalyzer()

def process_financial_report(pdf_file, text_input, user_question=""):
    """Main function to process financial report from PDF or text - IMPROVED ERROR HANDLING"""
    
    financial_text = ""
    
    # Process PDF if uploaded
    if pdf_file is not None:
        print("📄 Processing PDF file...")
        try:
            financial_text = analyzer.extract_text_from_pdf(pdf_file)
            print(f"📄 PDF processing result: {len(financial_text) if not financial_text.startswith('❌') else 'ERROR'}")
        except Exception as e:
            financial_text = f"❌ Error processing PDF: {str(e)}"
            print(f"❌ PDF processing exception: {e}")
    
    # Use text input if no PDF or as fallback
    if not financial_text or financial_text.startswith("❌") or financial_text.startswith("⚠️"):
        if text_input.strip():
            financial_text = text_input
            print("📝 Using text input instead of PDF")
        else:
            if not financial_text:
                return "Please provide either a PDF file or text input.", ""
            # Return the error message from PDF processing
            return financial_text, ""
    
    # Always perform main analysis
    try:
        main_analysis = analyzer.analyze_financial_report(financial_text)
    except Exception as e:
        main_analysis = f"❌ Error during analysis: {str(e)}"
        print(f"❌ Analysis error: {e}")
    
    # Answer specific question if provided
    additional_response = ""
    if user_question.strip():
        try:
            additional_response = analyzer.answer_question(financial_text, user_question)
        except Exception as e:
            additional_response = f"❌ Error answering question: {str(e)}"
            print(f"❌ Question answering error: {e}")
    
    return main_analysis, additional_response

# Create Gradio interface
with gr.Blocks(title="Financial Report Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📊 Financial Report Analyzer (PDF + Multi-Model)
    
    This application analyzes financial reports from PDF files or text with multiple AI models:
    - **📄 PDF Support**: Automatic text extraction from financial reports
    - **🤖 Multiple Models**: Lightweight models for speed, Phi-4 for advanced analysis
    - **📈 Comprehensive Analysis**: Financial statements, cash flow, balance sheet
    
    The system automatically selects the best model based on your hardware.
    """)
    
    with gr.Row():
        gr.Markdown(analyzer.get_model_info())
    
    with gr.Row():
        with gr.Column(scale=1):
            # PDF Upload
            pdf_input = gr.File(
                label="📄 Upload PDF Financial Report",
                file_types=['.pdf'],
                file_count="single"
            )
            
            gr.Markdown("**OR**")
            
            # Text Input (alternative to PDF)
            text_input = gr.Textbox(
                label="📝 Paste Financial Report Text",
                placeholder="Paste your financial report text here if not using PDF...",
                lines=10,
                max_lines=15
            )
            
            user_question = gr.Textbox(
                label="❓ Additional Question (Optional)",
                placeholder="Ask any specific question about the financial report...",
                lines=2
            )
            
            analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 💡 Tips:
            - **PDF Format**: Upload annual reports, quarterly statements, or any financial PDF
            - **Text Format**: Copy-paste from financial documents
            - **Questions**: Ask about specific metrics, trends, or recommendations
            - **Model Selection**: System automatically chooses based on your hardware
            """)
        
        with gr.Column(scale=2):
            with gr.Tab("📈 Main Analysis"):
                main_output = gr.Textbox(
                    label="Financial Analysis Report",
                    lines=30,
                    max_lines=40,
                    show_copy_button=True
                )
            
            with gr.Tab("❓ Additional Q&A"):
                additional_output = gr.Textbox(
                    label="Answer to Your Question",
                    lines=15,
                    max_lines=25,
                    show_copy_button=True
                )
            
            with gr.Tab("📄 Extracted Text"):
                extracted_text = gr.Textbox(
                    label="Text Extracted from PDF",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )
    
    # Event handlers
    analyze_btn.click(
        fn=process_financial_report,
        inputs=[pdf_input, text_input, user_question],
        outputs=[main_output, additional_output]
    )
    
    # Show extracted text when PDF is uploaded - IMPROVED
    def show_extracted_text(pdf_file):
        if pdf_file is not None:
            try:
                return analyzer.extract_text_from_pdf(pdf_file)
            except Exception as e:
                return f"❌ Error extracting text: {str(e)}"
        return ""
    
    pdf_input.change(
        fn=show_extracted_text,
        inputs=[pdf_input],
        outputs=[extracted_text]
    )
    
    # Add example
    gr.Examples(
        examples=[
            [
                None,  # No PDF file in example
                """Company ABC Financial Report - Q3 2024
                
Revenue: $15.2M (up 12% YoY)
Gross Profit: $8.1M (53% margin)
Operating Expenses: $6.2M
Net Income: $1.9M (12.5% margin)

Cash Flow from Operations: $2.8M
Free Cash Flow: $1.4M
Cash and Equivalents: $5.2M

Total Assets: $25.8M
Total Liabilities: $12.3M
Shareholders' Equity: $13.5M
Current Ratio: 2.1
Debt-to-Equity: 0.31

Key Highlights:
- Strong revenue growth driven by new product launches
- Improved operational efficiency with reduced overhead costs
- Solid cash position supporting future investments
- Balanced capital structure with manageable debt levels""",
                "What are the key financial strengths and potential risks?"
            ]
        ],
        inputs=[pdf_input, text_input, user_question],
        label="📋 Example Financial Report"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
