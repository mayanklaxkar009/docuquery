# Deploying DocuQuery on Render (free)

The app is a Gradio web server with no PyTorch, so it fits Render's free web service (512 MB RAM).

## 1. Hugging Face token
1. Go to https://huggingface.co/settings/tokens and choose **Create new token**, type **Fine-grained**.
2. Tick **Make calls to Inference Providers** and create it. This token powers both the embeddings and the LLM.

Without a token the app still works: search is keyword-based and answers quote the documents.

## 2. Create the service
1. Push this repo to GitHub.
2. In Render choose **New → Blueprint**, pick the repo and apply. `render.yaml` creates a free Python web service.
3. When asked for `HF_TOKEN`, paste the token (or add it later under the service's **Environment** tab).
4. The first build takes a few minutes. The service URL looks like `https://docuquery.onrender.com`.

Every push to `main` redeploys automatically. GitHub Actions runs the tests on each push.

## Troubleshooting
- **First request is slow.** Free Render services sleep after 15 minutes idle and take about a minute to wake.
- **Footer says "keyword matching".** `HF_TOKEN` is missing or the embedding API was unreachable at start-up; set the token and redeploy.
- **Answers are quotes, not prose.** The LLM call failed (usually token permissions or exhausted free credits). Try another model with the `LLM_MODEL` variable, e.g. `meta-llama/Llama-3.2-3B-Instruct`.
