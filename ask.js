document.addEventListener('DOMContentLoaded', () => {
  if (!localStorage.getItem('loggedInUser')) {
    window.location.href = 'index.html';
    return;
  }
  const chatMain = document.getElementById('chat-main');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const modelSelect = document.getElementById('model-select');
  const spinner = document.getElementById('chat-spinner');

  const urls = ['http://127.0.0.1:5000', 'http://localhost:5000'];

  const messages = [
    { role: 'system', content: 'You are a helpful assistant specialized in finance and general knowledge. Keep answers concise and accurate.' }
  ];

  function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = `msg ${role === 'user' ? 'user' : 'assistant'}`;
    div.textContent = content;
    chatMain.appendChild(div);
    chatMain.scrollTop = chatMain.scrollHeight;
  }

  function showSpinner() {
    if (spinner) spinner.style.display = 'flex';
  }
  function hideSpinner() {
    if (spinner) spinner.style.display = 'none';
  }

  function renderDetailCard(payload) {
    if (!payload) return;
    const { info, predictions, chart_image, ranking_image } = payload;
    const card = document.createElement('div');
    card.className = 'detail-card';
    const latestClose = info && typeof info.latest_close === 'number' ? info.latest_close.toFixed(2) : '-';
    const dateStr = info && info.latest_date ? info.latest_date : '-';
    let rows = '';
    if (Array.isArray(predictions)) {
      rows = predictions.map(p => `<tr><td>${p.name}</td><td style="text-align:right;">${Number(p.value).toFixed(2)}</td></tr>`).join('');
    }
    card.innerHTML = `
      <div class="detail-grid">
        <div class="kpi"><div class="label">Symbol</div><div class="value">${(info && info.symbol) || '-'}</div></div>
        <div class="kpi"><div class="label">Latest Close</div><div class="value">${latestClose}</div></div>
        <div class="kpi"><div class="label">Date</div><div class="value">${dateStr}</div></div>
      </div>
      <table class="pred-table"><thead><tr><th>Model</th><th style="text-align:right;">Predicted</th></tr></thead><tbody>${rows}</tbody></table>
    `;
    chatMain.appendChild(card);
    if (chart_image) {
      const img = document.createElement('img');
      img.src = chart_image; img.alt = 'Chart'; img.style.maxWidth = '100%'; img.style.marginTop = '6px';
      chatMain.appendChild(img);
    }
    if (ranking_image) {
      const img2 = document.createElement('img');
      img2.src = ranking_image; img2.alt = 'Ranking'; img2.style.maxWidth = '100%'; img2.style.marginTop = '6px';
      chatMain.appendChild(img2);
    }
    chatMain.scrollTop = chatMain.scrollHeight;
  }

  async function fetchJson(path, init) {
    let lastErr;
    for (const base of urls) {
      try {
        const resp = await fetch(`${base}${path}`, init);
        const text = await resp.text();
        let payload = {}; try { payload = JSON.parse(text); } catch {}
        if (!resp.ok) throw new Error(payload.error || `HTTP ${resp.status}`);
        return payload;
      } catch (e) { lastErr = e; }
    }
    throw lastErr || new Error('Network error');
  }

  let sending = false;
  async function send() {
    if (sending) return;
    const content = chatInput.value.trim();
    if (!content) return;
    addMessage('user', content);
    chatInput.value = '';
    sending = true;
    showSpinner();

    const selectedModel = modelSelect.value || 'deepseek-r1:latest';
    try {
      // First try Ollama chat
      const payload = await fetchJson('/ollama_chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel, messages: [...messages, { role: 'user', content }] })
      });
      const answer = payload.content || '(no response)';
      messages.push({ role: 'user', content });
      messages.push({ role: 'assistant', content: answer });
      addMessage('assistant', answer);
    } catch (e1) {
      // Fallback to built-in ML chatbot prediction
      try {
        const payload = await fetchJson('/chatbot_predict', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: content })
        });
        const answer = payload.content || '(no response)';
        messages.push({ role: 'user', content });
        messages.push({ role: 'assistant', content: answer });
        addMessage('assistant', answer);
        renderDetailCard(payload);
      } catch (e2) {
        addMessage('assistant', `Error: ${e1.message || e2.message}`);
      }
    }
    sending = false;
    hideSpinner();
  }

  sendBtn.addEventListener('click', send);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
});
