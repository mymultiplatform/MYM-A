import os
import requests
import base64
from datetime import datetime, timedelta
from dotenv import load_dotenv
from urllib.parse import quote
from flask import Flask, request

import os
from pathlib import Path
load_dotenv(r"C:\Users\cinco\Desktop\Repos\MYM-A-1\New year new me\Charlesschwab\.env")
# Get the directory containing the script
script_dir = Path(__file__).parent.resolve()
# Load .env from same directory as script
load_dotenv(os.path.join(script_dir, '.env'))
app = Flask(__name__)

class SchwabTraderAPI:
    def __init__(self):
        self.base_url = "https://api.schwabapi.com"
        self.client_id = os.getenv('SCHWAB_CLIENT_ID')
        self.client_secret = os.getenv('SCHWAB_CLIENT_SECRET')
        self.redirect_uri = os.getenv('SCHWAB_REDIRECT_URI')
        
        if not all([self.client_id, self.client_secret, self.redirect_uri]):
            raise ValueError("All OAuth credentials must be set in .env file")
        
        # Store tokens
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
    
    def get_authorization_url(self):
        """Generate the authorization URL for the OAuth flow"""
        return (f"{self.base_url}/v1/oauth/authorize?"
                f"client_id={self.client_id}&"
                f"redirect_uri={quote(self.redirect_uri)}")
    
    def get_basic_auth_header(self):
        """Generate Basic Auth header from client credentials"""
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    def exchange_code_for_tokens(self, authorization_code):
        """Exchange authorization code for access and refresh tokens"""
        headers = {
            'Authorization': self.get_basic_auth_header(),
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.post(
            f"{self.base_url}/v1/oauth/token",
            headers=headers,
            data=data
        )
        
        if response.ok:
            token_data = response.json()
            self._update_tokens(token_data)
            return token_data
        else:
            response.raise_for_status()
    
    def refresh_access_token(self):
        """Refresh the access token using the refresh token"""
        if not self.refresh_token:
            raise ValueError("No refresh token available")
        
        headers = {
            'Authorization': self.get_basic_auth_header(),
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token
        }
        
        response = requests.post(
            f"{self.base_url}/v1/oauth/token",
            headers=headers,
            data=data
        )
        
        if response.ok:
            token_data = response.json()
            self._update_tokens(token_data)
            return token_data
        else:
            response.raise_for_status()
    
    def _update_tokens(self, token_data):
        """Update token information from response"""
        self.access_token = token_data['access_token']
        self.refresh_token = token_data['refresh_token']
        self.token_expiry = datetime.now() + timedelta(seconds=token_data['expires_in'])
    
    def _ensure_valid_token(self):
        """Ensure we have a valid access token"""
        if not self.access_token:
            raise ValueError("No access token available. Please authenticate first.")
        
        if datetime.now() >= self.token_expiry - timedelta(minutes=1):
            self.refresh_access_token()
    
    def get_all_orders(self, max_results=100, from_date=None, to_date=None, status=None):
        """
        Get all orders across accounts with optional filtering
        """
        self._ensure_valid_token()
        
        endpoint = f"{self.base_url}/trader/v1/orders"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "maxResults": max_results
        }
        
        if from_date and to_date:
            params["fromEnteredTime"] = from_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            params["toEnteredTime"] = to_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        if status:
            params["status"] = status
        
        try:
            response = requests.get(
                endpoint,
                headers=headers,
                params=params
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching orders: {e}")
            return None

# Global variable to store the auth code
auth_code = None

@app.route('/')
@app.route('/callback')
def callback():
    global auth_code
    auth_code = request.args.get('code')
    if auth_code:
        return "Authorization code received! You can close this window and return to the console."
    return "Waiting for authorization code..."

def main():
    client = SchwabTraderAPI()
    
    print("\nPlease visit this URL in your browser to authorize the application:")
    auth_url = client.get_authorization_url()
    print(auth_url)
    
    print("\nStarting local server to receive callback...")
    from threading import Thread
    server = Thread(target=lambda: app.run(port=8080))
    server.daemon = True
    server.start()
    
    print("\nWaiting for authorization...")
    while auth_code is None:
        pass
    
    print("\nAuthorization code received!")
    
    try:
        # Exchange authorization code for tokens
        token_data = client.exchange_code_for_tokens(auth_code)
        print("Successfully obtained access token!")
        
        # Example: Get orders from the last 7 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        orders = client.get_all_orders(
            max_results=100,
            from_date=start_date,
            to_date=end_date
        )
        
        if orders:
            print("\nOrders found:")
            for order in orders:
                print("\nOrder Details:")
                print(f"Order ID: {order.get('orderId')}")
                print(f"Status: {order.get('status')}")
                print(f"Order Type: {order.get('orderType')}")
                print(f"Quantity: {order.get('quantity')}")
                
                if order.get('orderLegCollection'):
                    for leg in order['orderLegCollection']:
                        if 'instrument' in leg:
                            print(f"Symbol: {leg['instrument'].get('symbol')}")
                            print(f"Instruction: {leg.get('instruction')}")
        else:
            print("No orders found or error occurred")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()