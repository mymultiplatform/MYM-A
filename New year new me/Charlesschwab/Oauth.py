import os
import base64
import requests
import webbrowser
from loguru import logger

class SchwabAPIClient:
    def __init__(self):
        # CincoData Inc. credentials
        self.app_key = "XLx3yG8WArLDPak21kxAAAvbksAnzlu4"
        self.app_secret = "HF2XOTGGuxlwNljN"
        self.callback_url = "https://127.0.0.1"
        self.base_url = "https://api.schwabapi.com/v1"
        self.access_token = None
        self.refresh_token = None

    def construct_auth_url(self) -> str:
        """Constructs the initial authentication URL"""
        auth_url = (
            f"{self.base_url}/oauth/authorize"
            f"?client_id={self.app_key}"
            f"&redirect_uri={self.callback_url}"
        )
        logger.info("Authentication URL:")
        logger.info(auth_url)
        return auth_url

    def get_initial_tokens(self, returned_url: str):
        """Gets initial access and refresh tokens using the returned URL"""
        try:
            # Extract code from returned URL
            response_code = returned_url[returned_url.index('code=') + 5:].split('&')[0]
            
            # Create credentials
            credentials = f"{self.app_key}:{self.app_secret}"
            base64_credentials = base64.b64encode(credentials.encode()).decode()

            # Prepare headers and payload
            headers = {
                "Authorization": f"Basic {base64_credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            payload = {
                "grant_type": "authorization_code",
                "code": response_code,
                "redirect_uri": self.callback_url,
            }

            # Make token request
            response = requests.post(
                f"{self.base_url}/oauth/token",
                headers=headers,
                data=payload,
            )

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                logger.info("Successfully obtained tokens")
                return token_data
            else:
                logger.error(f"Failed to get tokens: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error getting tokens: {str(e)}")
            return None

    def refresh_tokens(self):
        """Refreshes the access token using the refresh token"""
        if not self.refresh_token:
            logger.error("No refresh token available")
            return None

        try:
            credentials = f"{self.app_key}:{self.app_secret}"
            base64_credentials = base64.b64encode(credentials.encode()).decode()

            headers = {
                "Authorization": f"Basic {base64_credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            payload = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            }

            response = requests.post(
                f"{self.base_url}/oauth/token",
                headers=headers,
                data=payload,
            )

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                logger.info("Successfully refreshed tokens")
                return token_data
            else:
                logger.error(f"Failed to refresh tokens: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error refreshing tokens: {str(e)}")
            return None

    def get_account_numbers(self):
        """Gets account numbers and their hash values"""
        if not self.access_token:
            logger.error("No access token available")
            return None

        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        response = requests.get(
            f"{self.base_url}/trader/v1/accounts/accountNumbers",
            headers=headers
        )

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get account numbers: {response.text}")
            return None

def main():
    # Initialize client
    client = SchwabAPIClient()
    
    # Get authentication URL and open in browser
    auth_url = client.construct_auth_url()
    webbrowser.open(auth_url)
    
    # Get the returned URL from user
    logger.info("After logging in, please paste the returned URL here:")
    returned_url = input()
    
    # Get initial tokens
    token_data = client.get_initial_tokens(returned_url)
    
    if token_data:
        logger.info("Authentication successful!")
        logger.debug(f"Token data: {token_data}")
        
        # Try to get account numbers as a test
        account_data = client.get_account_numbers()
        if account_data:
            logger.info("Successfully retrieved account numbers:")
            logger.debug(account_data)
    else:
        logger.error("Authentication failed!")

if __name__ == "__main__":
    main()