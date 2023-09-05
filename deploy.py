import subprocess
from pyngrok import ngrok
import os
from dotenv import load_dotenv
load_dotenv()

# Replace with your actual Ngrok authentication token
ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")

streamlit_port = 1234

subprocess.Popen(["streamlit", "run", "--server.port", str(streamlit_port), "app.py"])

# Open a tunnel to your Streamlit app
streamlit_tunnel = ngrok.connect(addr=streamlit_port, proto="http", bind_tls=True)
print("Streamlit URL:", streamlit_tunnel.public_url, flush=True)

try:
    input("Press Enter to exit...")
except KeyboardInterrupt:
    pass



