# Deploying DocuQuery (GitHub + Hugging Face Spaces, all free)

You need: a GitHub account, a Hugging Face account, and Git installed.
Time: about 30 minutes the first time. The first Space build takes 5 to 10 minutes.

## 1. Run it on your own machine first

1. Unzip the project and open a terminal in the `docuquery` folder.
2. Create a virtual environment and install everything:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```
3. Run the tests: `pytest -q`. You should see 30 passed.
4. Start the app: `python app.py`, open http://127.0.0.1:7860 and click **Try the sample**.
   Without a token, answers are extractive (quoted sentences). That's expected.
5. Run the evaluation: `python scripts/evaluate_retrieval.py`. Note the hit rate and MRR; you can quote them on your CV.

## 2. Create two Hugging Face tokens

Go to https://huggingface.co/settings/tokens.

1. **Inference token** (used by the app): New token → Fine-grained → tick
   "Make calls to Inference Providers". Name it `docuquery-inference`.
2. **Deploy token** (used by GitHub Actions): New token → Write. Name it `github-deploy`.

Keep both somewhere safe. Never commit a token to Git.

## 3. Create the Space

1. https://huggingface.co/new-space
2. Name: `docuquery`. Licence: MIT. SDK: **Gradio**. Hardware: **CPU basic (free)**. Visibility: Public.
3. Open the Space → **Settings** → **Variables and secrets** → **New secret**:
   name `HF_TOKEN`, value = your inference token.

## 4. Push the code to GitHub

1. Create an empty public repo at https://github.com/new named `docuquery` (no README, no licence; the project has both).
2. In the project folder:
   ```bash
   git init -b main
   git add .
   git commit -m "DocuQuery: RAG question answering over PDFs"
   git remote add origin https://github.com/YOUR-GITHUB-USERNAME/docuquery.git
   git push -u origin main
   ```

## 5. Connect GitHub to the Space (automatic deploys)

In the GitHub repo → **Settings** → **Secrets and variables** → **Actions**:

1. **Secrets** tab → New repository secret: `HF_TOKEN` = your **deploy** (write) token.
2. **Variables** tab → New repository variable: `HF_SPACE` = `mayanklaxkar008/docuquery`.

Then go to the **Actions** tab → CI → **Run workflow** (or push any commit).
The `test` job runs pytest; if it passes, `deploy` pushes the code to your Space.

## 6. Check the live app

Open `https://huggingface.co/spaces/mayanklaxkar008/docuquery`. The **Logs** tab shows the build.
When it says Running, click **Try the sample** and ask: *How is retrieval quality measured?*
The answer should now be written by the LLM, with [1]-style citations.

## 7. Finish the README

Replace `YOUR-HF-USERNAME` and `YOUR-GITHUB-USERNAME` in `README.md`, commit and push.
CI redeploys automatically.

## Troubleshooting

- **"Inference Providers" error or fallback note on every answer.** The token lacks the inference permission,
  the secret name isn't exactly `HF_TOKEN`, or the free monthly credit is used up.
  You can also try another model: add a Space variable `LLM_MODEL`, for example `meta-llama/Llama-3.2-3B-Instruct`
  (some Llama models need you to accept their licence on the model page first).
- **Build fails on gradio version.** Change `sdk_version` in the README header to the version the Space settings suggest.
- **Push to the Space rejected for binary files.** Don't commit PDFs or images; the sample PDF is generated at start-up.
- **Space goes to sleep.** Free Spaces sleep after inactivity and wake on the next visit (about a minute).
