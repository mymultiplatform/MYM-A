from playwright.sync_api import sync_playwright
import time
from fpdf import FPDF
import os

def create_pdf_from_texts(contents_dict, output_file="pytorch_docs.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for url, content in contents_dict.items():
        # Add a new page for each URL
        pdf.add_page()
        
        # Set font for URL header
        pdf.set_font('Helvetica', 'B', 16)
        
        # Add URL as header
        pdf.cell(0, 10, "Source:", ln=True)
        pdf.set_font('Helvetica', '', 12)
        pdf.multi_cell(0, 10, url)
        pdf.ln(10)
        
        # Add separator line
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
        
        # Set font for content
        pdf.set_font('Helvetica', '', 12)
        
        # Split content into smaller chunks to avoid buffer overflow
        chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
        for chunk in chunks:
            try:
                pdf.multi_cell(0, 10, chunk)
            except Exception as e:
                print(f"Warning: Could not write chunk, skipping. Error: {str(e)}")
                continue
    
    try:
        pdf.output(output_file)
        print(f"PDF successfully created: {output_file}")
    except Exception as e:
        print(f"Error saving PDF: {str(e)}")
        raise

def scrape_pytorch_docs():
    urls = [
        "https://pytorch.org/docs/stable/torch.html",
        "https://pytorch.org/docs/stable/nn.html",
        "https://pytorch.org/docs/stable/nn.functional.html",
        "https://pytorch.org/docs/stable/tensors.html",
        "https://pytorch.org/docs/stable/tensor_attributes.html",
        "https://pytorch.org/docs/stable/tensor_view.html",
        "https://pytorch.org/docs/stable/amp.html",
        "https://pytorch.org/docs/stable/autograd.html",
        "https://pytorch.org/docs/stable/library.html",
        "https://pytorch.org/docs/stable/cpu.html",
        "https://pytorch.org/docs/stable/cuda.html",
        "https://pytorch.org/docs/stable/torch_cuda_memory.html",
        "https://pytorch.org/docs/stable/torch_cuda_memory.html#generating-a-snapshot",
        "https://pytorch.org/docs/stable/torch_cuda_memory.html#using-the-visualizer",
        "https://pytorch.org/docs/stable/torch_cuda_memory.html#snapshot-api-reference",
        "https://pytorch.org/docs/stable/export.html",
        "https://pytorch.org/docs/stable/distributed.html"
    ]
    
    contents_dict = {}
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        for url in urls:
            try:
                print(f"Processing: {url}")
                page.goto(url, wait_until="networkidle")
                time.sleep(2)  # Allow time for dynamic content to load
                
                # Get all text content from the page
                content = page.evaluate('''() => document.body.innerText''')
                contents_dict[url] = content
                
                print(f"Successfully scraped content from: {url}")
                
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                contents_dict[url] = f"Error processing this URL: {str(e)}"
        
        browser.close()
    
    return contents_dict

def main():
    print("Starting documentation scraping...")
    contents_dict = scrape_pytorch_docs()
    
    print("Creating PDF...")
    create_pdf_from_texts(contents_dict)
    
    print("Process completed! Check pytorch_docs.pdf")

if __name__ == "__main__":
    main()