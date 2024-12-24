import os
import subprocess

# CHANGE THIS PATH TO YOUR NOTEBOOK
notebook_path = r"C:\Users\cinco\Desktop\MYM-A-1\collabbuild.ipynb"

def convert_to_pdf():
    try:
        # First convert to HTML (more reliable than direct PDF conversion)
        cmd = f'jupyter nbconvert --to html "{notebook_path}"'
        subprocess.run(cmd, shell=True)
        
        # Get the HTML file path
        html_path = notebook_path.replace('.ipynb', '.html')
        
        # Use Chrome or Edge to print to PDF (if available)
        # Try Chrome first
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
        ]
        
        edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        
        browser_path = None
        for path in chrome_paths:
            if os.path.exists(path):
                browser_path = path
                break
                
        if not browser_path and os.path.exists(edge_path):
            browser_path = edge_path
            
        if browser_path:
            pdf_path = notebook_path.replace('.ipynb', '.pdf')
            cmd = f'"{browser_path}" --headless --disable-gpu --print-to-pdf="{pdf_path}" "{html_path}"'
            subprocess.run(cmd, shell=True)
            print(f"PDF created at: {pdf_path}")
            
            # Clean up HTML file
            os.remove(html_path)
        else:
            print("Could not find Chrome or Edge browser. Please install one of them.")
            
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    print("Converting notebook to PDF...")
    convert_to_pdf()