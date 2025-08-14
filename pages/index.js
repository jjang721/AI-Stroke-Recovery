import { useState } from "react";

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!query.trim()) return;
    const newMessages = [...messages, { role: "user", content: query }];
    setMessages(newMessages);
    setQuery("");
    setLoading(true);

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: query }),
    });

    const data = await res.json();
    setMessages([...newMessages, { role: "assistant", content: data.answer }]);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-yellow-100 font-sans p-6">
      <h1 className="text-3xl text-center font-bold text-gray-800 mb-4">
        🧠 Stroke Recovery Assistant
      </h1>
      <div className="max-w-2xl mx-auto space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-3 rounded-lg shadow-md max-w-[80%] ${
              msg.role === "user" ? "bg-green-100 ml-auto" : "bg-white mr-auto border"
            }`}
          >
            <strong>{msg.role === "user" ? "You" : "AI"}:</strong> {msg.content}
          </div>
        ))}
        <div className="flex gap-2 mt-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="💬 Enter your question..."
            className="flex-1 p-2 border rounded-lg"
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg disabled:bg-gray-400"
          >
            {loading ? "Thinking..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
