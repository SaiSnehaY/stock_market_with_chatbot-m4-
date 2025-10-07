'use strict';

document.addEventListener('DOMContentLoaded', () => {
  if (!localStorage.getItem('loggedInUser')) {
    window.location.href = 'index.html';
    return;
  }

  // Persist last payload for chart switching
  let lastPayload = null;

  function clearChartView() {
    if (chartView) {
      chartView.innerHTML = '';
    }
  }

  function hasOHLC(hist) {
    return Array.isArray(hist) && hist.length && hist[0].Open != null && hist[0].High != null && hist[0].Low != null && hist[0].Close != null;
  }

  function setChartOptionsAvailability(payload, info) {
    if (!chartSelect) return;
    const opts = {
      line: chartSelect.querySelector('option[value="line"]'),
      hist: chartSelect.querySelector('option[value="hist"]'),
      range: chartSelect.querySelector('option[value="range"]'),
      candle: chartSelect.querySelector('option[value="candle"]'),
    };
    const histArr = payload && payload.historical_prices;
    const lineOk = Array.isArray(histArr) && histArr.length > 1;
    const histOk = lineOk;
    const rangeOk = info && info.day_low != null && info.day_high != null && info.latest_close != null;
    const candleOk = hasOHLC(histArr);
    if (opts.line) opts.line.disabled = !lineOk;
    if (opts.hist) opts.hist.disabled = !histOk;
    if (opts.range) opts.range.disabled = !rangeOk;
    if (opts.candle) opts.candle.disabled = !candleOk;
    // Select first available if current is disabled
    if (chartSelect.options[chartSelect.selectedIndex]?.disabled) {
      const firstEnabled = Array.from(chartSelect.options).find(o => !o.disabled);
      if (firstEnabled) chartSelect.value = firstEnabled.value;
    }
  }

  function renderChart(type, payload, info) {
    if (!chartView) return;
    clearChartView();
    const div = document.createElement('div');
    chartView.appendChild(div);
    const hist = payload && payload.historical_prices;
    if (type === 'line' && Array.isArray(hist) && hist.length) {
      const dates = hist.map(p => p.Date);
      const closes = hist.map(p => Number(p.Close));
      Plotly.newPlot(div, [{ x: dates, y: closes, type: 'scatter', mode: 'lines', line: { color: '#8b5cf6' } }], {
        margin: { l: 30, r: 10, t: 20, b: 30 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', yaxis: { title: 'Price' }, xaxis: { title: 'Date' }
      }, { displayModeBar: false });
      return;
    }
    if (type === 'hist' && Array.isArray(hist) && hist.length) {
      const closes = hist.map(p => Number(p.Close));
      Plotly.newPlot(div, [{ x: closes, type: 'histogram', marker: { color: '#a78bfa' }, nbinsx: 15 }], {
        margin: { l: 30, r: 10, t: 20, b: 30 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', xaxis: { title: 'Price' }, yaxis: { title: 'Frequency' }
      }, { displayModeBar: false });
      return;
    }
    if (type === 'range' && info && info.day_low != null && info.day_high != null && info.latest_close != null) {
      const low = Number(info.day_low), high = Number(info.day_high), close = Number(info.latest_close);
      const mid = (low + high) / 2;
      Plotly.newPlot(div,
        [
          { x: [low, high], y: ['Today'], type: 'scatter', mode: 'lines', line: { color: '#a78bfa', width: 8 }, showlegend: false },
          { x: [close], y: ['Today'], type: 'scatter', mode: 'markers', marker: { color: '#ef4444', size: 10 }, name: 'Latest' },
          { x: [mid], y: ['Today'], type: 'scatter', mode: 'markers', marker: { color: '#10b981', size: 8 }, name: 'Mid' }
        ],
        { margin: { l: 50, r: 10, t: 10, b: 30 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', xaxis: { title: 'Price' }, yaxis: { showticklabels: false } },
        { displayModeBar: false }
      );
      return;
    }
    if (type === 'candle' && hasOHLC(hist)) {
      const o = hist.map(p => Number(p.Open));
      const h = hist.map(p => Number(p.High));
      const l = hist.map(p => Number(p.Low));
      const c = hist.map(p => Number(p.Close));
      const d = hist.map(p => p.Date);
      Plotly.newPlot(div, [{ type: 'candlestick', x: d, open: o, high: h, low: l, close: c, increasing: { line: { color: '#10b981' } }, decreasing: { line: { color: '#ef4444' } } }], {
        margin: { l: 30, r: 10, t: 20, b: 30 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent'
      }, { displayModeBar: false });
      return;
    }
    // Fallback message
    div.innerHTML = '<div style="color:#6b7280; padding:8px;">No data for the selected chart.</div>';
  }

  const chatMain = document.getElementById('chat-main');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const modelSelect = document.getElementById('model-select');
  const spinner = document.getElementById('chat-spinner');
  const chartSelect = document.getElementById('chart-select');
  const chartView = document.getElementById('chart-view');

  const urls = ['http://127.0.0.1:5000', 'http://localhost:5000'];

  function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = `msg ${role === 'user' ? 'user' : 'assistant'}`;
    div.textContent = content;
    chatMain.appendChild(div);
    chatMain.scrollTop = chatMain.scrollHeight;
  }
  function showSpinner(){ if (spinner) spinner.style.display = 'flex'; }
  function hideSpinner(){ if (spinner) spinner.style.display = 'none'; }

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

  function extractSymbolFrom(text) {
    const t = (text || '').trim();
    const direct = t.match(/\b[A-Z]{1,5}\b/);
    if (direct) return direct[0];
    const map = {
      apple: 'AAPL', microsoft: 'MSFT', alphabet: 'GOOGL', google: 'GOOGL', amazon: 'AMZN',
      tesla: 'TSLA', meta: 'FB', nvidia: 'NVDA', jpmorgan: 'JPM', visa: 'V', pg: 'PG',
    };
    const lower = t.toLowerCase();
    for (const k of Object.keys(map)) {
      if (lower.includes(k)) return map[k];
    }
    return null;
  }

  async function handleGeminiQuery(query) {
    // For now, GEMINI model = fetch live stock details via backend /stock_info
    // Behavior: parse a symbol from the query (or treat the whole input as a ticker), fetch details and show a concise answer
    let symbol = extractSymbolFrom(query) || query.trim().toUpperCase();
    if (!/^[A-Z]{1,5}$/.test(symbol)) {
      symbol = 'AAPL';
    }
    const payload = await fetchJson('/stock_info', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stock_symbol: symbol, stock: symbol })
    });
    const info = payload.info || {};
    const lines = [
      `Symbol: ${info.symbol || symbol}`,
      `Name: ${info.name || symbol}`,
      `Latest Close: ${info.latest_close != null ? info.latest_close : '-'}`,
      `Date: ${info.latest_date || '-'}`,
      `Day High: ${info.day_high != null ? info.day_high : '-'}`,
      `Day Low: ${info.day_low != null ? info.day_low : '-'}`,
      `Volume: ${info.volume != null ? info.volume : '-'}`,
    ];
    return { text: lines.join('\n'), payload };
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

    const model = (modelSelect.value || 'gemini');
    try {
      if (model === 'gemini') {
        const { text, payload } = await handleGeminiQuery(content);
        addMessage('assistant', text);
        // Render a small details card
        const d = document.createElement('div');
        d.className = 'detail-card';
        const info = payload.info || {};
        d.innerHTML = `
          <div class="detail-grid">
            <div class="kpi"><div class="label">Symbol</div><div class="value">${info.symbol || '-'}</div></div>
            <div class="kpi"><div class="label">Latest Close</div><div class="value">${info.latest_close != null ? Number(info.latest_close).toFixed(2) : '-'}</div></div>
            <div class="kpi"><div class="label">Date</div><div class="value">${info.latest_date || '-'}</div></div>
          </div>
        `;
        chatMain.appendChild(d);

        // Prepare chart selector and render single chart view
        lastPayload = payload;
        setChartOptionsAvailability(payload, info);
        if (typeof window.enterSplitMode === 'function') window.enterSplitMode();
        const currentType = chartSelect ? chartSelect.value : 'line';
        renderChart(currentType, payload, info);
        if (chartSelect && !chartSelect._wired) {
          chartSelect.addEventListener('change', () => {
            const type = chartSelect.value;
            renderChart(type, lastPayload, info);
          });
          chartSelect._wired = true;
        }

        chatMain.scrollTop = chatMain.scrollHeight;
      } else {
        addMessage('assistant', 'Only Gemini is supported in this chat.');
      }
    } catch (e) {
      addMessage('assistant', `Error: ${e.message}`);
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
