// pages/api/chat.js

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }
  const { question } = req.body;
  // TODO: Connect to your Python backend, OpenAI, or other service here.
  // For now, return a placeholder response.
  res.status(200).json({ answer: `You asked: ${question}. (Backend integration needed)` });
}
