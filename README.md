# tortally

1. Make an *ngrok* account and get your `NGROK_AUTH_TOKEN`.
2. Change `.sample.env` to `.env`, and fill in your token generated and your OpenAI token.
3. Install [ngrok](https://ngrok.com/).
3. Install the project dependencies: `poetry install`
4. Run the project: `poetry run python deploy.py`