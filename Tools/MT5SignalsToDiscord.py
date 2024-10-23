from discord_webhook import DiscordWebhook

# Function to send the message to Discord
def send_to_discord(message):
    webhook_url = 'https://discord.com/api/webhooks/1298744394324639804/eY5xzTrrXoBaYfOsCgdx1UC8Lf-Tvy1uVnSc40JwOHHS4YTnUz1sUBpuvgTdHJZNmatd'
    webhook = DiscordWebhook(url=webhook_url, content=message)
    response = webhook.execute()

# Example usage
if __name__ == "__main__":
    # For now, we'll just send a simple message
    daily_news = "Here's your daily brief news summary!"
    send_to_discord(daily_news)
