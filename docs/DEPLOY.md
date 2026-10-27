# Deploying DocuQuery for free (Render + Groq)

Everything here runs on free tiers with no credit card:
- **Render free web service** (512 MB RAM) hosts the Gradio app.
- **BM25 retrieval** runs inside the app, so no embedding model or API is needed.
- **Groq free API** runs the LLM (`openai/gpt-oss-120b` by default).

## 1. Groq API key (for LLM answers)
1. Sign in at https://console.groq.com and open **API Keys**.
2. Choose **Create API Key**, name it `docuquery`, and copy it (it starts with `gsk_`).

Without a key the app still works: answers quote the most relevant sentences from the PDF.

## 2. Render service
1. In Render choose **New → Web Service → Public Git Repository** and paste the GitHub URL
   (or **New → Blueprint** to use `render.yaml` and get auto-deploys on push).
2. Build command `pip install -r requirements.txt`, start command `python app.py`, plan **Free**.
3. Under **Environment** add `PYTHON_VERSION=3.11.9` and `GROQ_API_KEY=<your key>`.

## 3. Keep it awake
`.github/workflows/keepalive.yml` pings the app every 10 minutes so recruiters don't wait for a cold start.
If your URL differs, set a repository variable `DEMO_URL` (Settings → Secrets and variables → Actions → Variables).
GitHub pauses scheduled workflows after 60 days without commits; re-enable it from the Actions tab.

## Troubleshooting
- **Footer says "extractive mode".** `GROQ_API_KEY` is missing; add it and redeploy.
- **Answers suddenly become quotes.** Groq's free rate limit was hit (per minute or per day); it recovers on its own.
- **Use another provider.** Set `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` for any OpenAI-compatible API.
