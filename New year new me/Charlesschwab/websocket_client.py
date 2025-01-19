# websocket_client.py
import websocket
import json
from datetime import datetime

class SchwabWebSocket:
    def __init__(self):
        self.ws = None
        
        # Load token
        with open('token.json', 'r') as f:
            self.token = json.load(f)
            
    def connect(self):
        headers = {
            "Authorization": f"Bearer {self.token['access_token']}",
            "Content-Type": "application/json"
        }
        
        self.ws = websocket.WebSocketApp(
            "wss://api.schwabapi.com/marketdata/v1/ws",
            header=headers,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        self.ws.run_forever()

    def on_message(self, ws, message):
        print(f"Received: {message}")

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"Connection closed: {close_msg}")

    def on_open(self, ws):
        print("Connection opened")
        self.subscribe_to_options()

    def subscribe_to_options(self):
        subscribe_msg = {
            "requests": [{
                "service": "LEVELONE_OPTIONS",
                "requestid": "1",
                "command": "SUBS",
                "parameters": {
                    "keys": "AAPL_250117C200000",  # Example option
                    "fields": "0,1,2,3,4,5,28,29,30,31,32"
                }
            }]
        }
        self.ws.send(json.dumps(subscribe_msg))