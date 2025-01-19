import pandas as pd

def get_csv_datatypes(file_path):
    """
    Read a CSV file and return the data types for each column.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary containing column names and their corresponding data types
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Create a dictionary of column names and their data types
        datatypes = {col: str(df[col].dtype) for col in df.columns}
        
        return datatypes
        
    except FileNotFoundError:
        return "Error: File not found"
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage:
datatypes = get_csv_datatypes('/Users/jazzhashzzz/Desktop/data for scripts/data bento data/SPY/SPY')

for column, dtype in datatypes.items():
     print(f"{column}: {dtype}")