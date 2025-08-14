StrokeRehab.ai is an AI-driven rehabilitation assistant designed to provide personalized, evidence-based stroke recovery guidance. It leverages LangChain, OpenAI embeddings, and a Chroma vector database to retrieve relevant insights from over 50+ peer-reviewed scientific documents.

This project bridges cutting-edge NLP with healthcare accessibility, enabling stroke survivors and caregivers to access trusted recovery information in real-time.

Features
1. AI Chat Interface – Conversational assistant for stroke recovery questions

2. Retrieval-Augmented Generation (RAG) – Answers grounded in scientific literature

3. Real-Time Response – Optimized retrieval pipeline for minimal latency

4. Source-Aware Citations – Provides document sources for transparency

5. Deployed Web App – Built with Streamlit for accessibility

Tech Stack

Layer	Tools & Frameworks
Streamlit
Python, LangChain
OpenAI API (Embeddings + Chat Models)
Chroma Vector DB
Streamlit Cloud / Vercel
Git, GitHub

---

## Next.js React Chat App

This directory now contains a minimal Next.js-compatible React chat app for deployment on Vercel.

- `pages/index.js`: Main chat UI
- `pages/api/chat.js`: API route for chat backend (stubbed for now)

### How to use
1. If you haven't already, initialize a Next.js project here:
   ```sh
   npx create-next-app@latest . --js --tailwind --eslint --app --src-dir
   ```
2. Place the provided `pages/index.js` and `pages/api/chat.js` files in the correct locations.
3. Run locally:
   ```sh
   npm run dev
   ```
4. Deploy to Vercel for production.

---
