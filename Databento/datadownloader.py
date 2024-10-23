import databento as db
import os
from dotenv import load_dotenv
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("API_KEY")

load_dotenv(Path("Databento/apikey.env"))
print(os.getenv("DATABENTO_API_KEY"))



def download_databento_data(job_id: str, output_dir: str = "databento_data"):
    """
    Download data from Databento using a job ID.
    
    Args:
        job_id (str): The Databento job ID
        output_dir (str): Directory where the files should be saved
    """
    try:
        # Create client
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY not found in environment variables")
            
        client = db.Historical(api_key)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # List all files available for the job
        files = client.batch.list_files(job_id)
        print(f"Found {len(files)} files to download")
        
        # Download all files
        downloaded_files = client.batch.download(
            job_id=job_id,
            output_dir=output_path
        )
        
        print(f"Successfully downloaded {len(downloaded_files)} files to {output_path}")
        return downloaded_files
        
    except db.BentoClientError as e:
        print(f"Client error: {e}")
    except db.BentoServerError as e:
        print(f"Server error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Your job ID from the screenshot
    JOB_ID = "XNAS-20241022-PAMBJ3M65M"
    
    # Download the data
    downloaded_files = download_databento_data(JOB_ID)
    
    # Print downloaded files
    if downloaded_files:
        print("\nDownloaded files:")
        for file in downloaded_files:
            print(f"- {file}")