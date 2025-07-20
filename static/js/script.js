async function submitQuestion() {
    const question = document.getElementById("question").value;
    const api_key = document.getElementById("api_key").value;
    const responseBox = document.getElementById("responseBox");
    const loading = document.getElementById("loading");

    responseBox.innerText = "";

    if (!question.trim() || !api_key.trim()) {
        responseBox.innerText = "⚠️ Please enter both a question and your Groq API key.";
        return;
    }

    loading.style.display = "block";

    try {
        const res = await fetch("/solve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, api_key })
        });

        const data = await res.json();
        if (data.response) {
            responseBox.innerText = "✅ " + data.response;
        } else {
            responseBox.innerText = "❌ Error: " + (data.error || "Unknown error");
        }
    } catch (err) {
        responseBox.innerText = "❌ Failed to connect to server.";
    }

    loading.style.display = "none";
}