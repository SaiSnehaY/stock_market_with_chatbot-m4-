document.addEventListener('DOMContentLoaded', () => {
    const logoutButton = document.getElementById('logout-button');
    const stockSelect = document.getElementById('stock-select');
    const dateInput = document.getElementById('date-input');
    const predictButton = document.getElementById('predict-button');
    const chartTypeSelect = document.getElementById('chart-type');
    const predictionsBody = document.getElementById('predictions-body');
    const controlsCard = document.getElementById('controls-card');
    const themeToggle = document.getElementById('theme-toggle');
    const exportDocBtn = document.getElementById('export-doc');
    let lastPayloadForReport = null;
    // AI modal elements
    const aiBtn = document.getElementById('ai-predictor-btn');
    const aiModal = document.getElementById('ai-modal');
    const aiModalBackdrop = document.getElementById('ai-modal-backdrop');
    const aiClose = document.getElementById('ai-modal-close');
    const aiStockSelect = document.getElementById('ai-stock-select');
    const aiDateInput = document.getElementById('ai-date-input');
    const aiChartType = document.getElementById('ai-chart-type');
    const aiVisualization = document.getElementById('ai-visualization');
    const aiModel = document.getElementById('ai-model');
    const aiSubmit = document.getElementById('ai-submit');
    const aiExportDocBtn = document.getElementById('ai-export-doc');
    const aiSelectedModelEl = document.getElementById('ai-selected-model');
    const aiPredictedPriceEl = document.getElementById('ai-predicted-price');
    const aiChartContainer = document.getElementById('ai-chart-container');
    const aiRankingContainer = document.getElementById('ai-ranking-container');
    let aiChartImgEl = null;
    let aiChartCaptionEl = null;
    let aiLastPayloadForReport = null;

    const stockNameEl = document.getElementById('stock-name');
    const sumSymbol = document.getElementById('sum-symbol');
    const sumClose = document.getElementById('sum-close');
    const sumDate = document.getElementById('sum-date');
    const sumHigh = document.getElementById('sum-high');
    const sumLow = document.getElementById('sum-low');
    const sumVolume = document.getElementById('sum-volume');
    const kpiPrice = document.getElementById('kpi-price');
    const kpiChange = document.getElementById('kpi-change');
    const kpiVolume = document.getElementById('kpi-volume');
    const sparkline = document.getElementById('sparkline');

    const resultsCard = document.getElementById('results-card');
    const chartContainer = document.getElementById('chart-container');
    const rankingContainer = document.getElementById('ranking-container');
    const compareButton = document.getElementById('compare-button');
    const visualizationBtn = document.getElementById('visualization-btn');
    const askAnythingTop = document.getElementById('ask-anything-top');
    const welcomeMessage = document.getElementById('welcome-message');
    const userGreeting = document.getElementById('user-greeting');
    let chartImgEl = null;
    let chartCaptionEl = null;

    if (!localStorage.getItem('loggedInUser')) {
        window.location.href = 'index.html';
    }

    // Apply saved theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') document.body.classList.add('dark');
    themeToggle.textContent = savedTheme === 'dark' ? 'Light Theme' : 'Dark Theme';

    // Hamburger menu toggle
    const menuButton = document.getElementById('menu-button');
    const extrasPanel = document.getElementById('extras-panel');
    
    menuButton.addEventListener('click', () => {
        extrasPanel.classList.toggle('show');
    });

    // Ask Anything (top-right button) -> navigate to new chat page
    if (askAnythingTop) {
        askAnythingTop.addEventListener('click', () => {
            window.location.href = 'chat.html';
        });
    }

    // Close extras panel when clicking outside
    document.addEventListener('click', (e) => {
        if (!menuButton.contains(e.target) && !extrasPanel.contains(e.target)) {
            extrasPanel.classList.remove('show');
        }
    });

    async function fetchJson(path, init) {
        const urls = [`http://127.0.0.1:5000${path}`, `http://localhost:5000${path}`];
        let lastErr;
        for (const url of urls) {
            try {
                const resp = await fetch(url, init);
                const text = await resp.text();
                let payload = {}; try { payload = JSON.parse(text); } catch {}
                if (!resp.ok) throw new Error(payload.error || `HTTP ${resp.status}`);
                return payload;
            } catch (e) { lastErr = e; }
        }
        throw lastErr || new Error('Network error');
    }

    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, '0');
    const dd = String(today.getDate()).padStart(2, '0');
    dateInput.value = `${yyyy}-${mm}-${dd}`;
    if (aiDateInput) aiDateInput.value = `${yyyy}-${mm}-${dd}`;

    logoutButton.addEventListener('click', () => {
        localStorage.removeItem('loggedInUser');
        window.location.href = 'index.html';
    });

    themeToggle.addEventListener('click', () => {
        const isDark = document.body.classList.toggle('dark');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        themeToggle.textContent = isDark ? 'Light Theme' : 'Dark Theme';
    });

    // Compare button
    compareButton.addEventListener('click', () => {
        window.location.href = 'compare.html';
    });

    // Set personalized greeting
    const loggedInUser = localStorage.getItem('loggedInUser');
    if (loggedInUser) {
        welcomeMessage.textContent = `Hi ${loggedInUser}!`;
        userGreeting.textContent = 'Welcome to Stock Price Prediction System';
    }

    // Visualization button
    visualizationBtn.addEventListener('click', () => {
        if (!lastPayloadForReport) {
            alert('Please run a prediction first to view visualizations.');
            return;
        }
        
        // Save data to session storage
        const visualizationData = {
            rows: lastPayloadForReport.rows,
            info: lastPayloadForReport.info
        };
        sessionStorage.setItem('visualizationData', JSON.stringify(visualizationData));
        
        // Redirect to visualization page
        window.location.href = 'visualization.html';
    });

    // (Ask Anything removed)



    // AI Modal open/close
    function openAiModal() {
        if (!aiModal) return;
        aiModal.style.display = 'block';
        aiModal.setAttribute('aria-hidden', 'false');
        // Default selections mirror main controls for convenience
        if (stockSelect && aiStockSelect && !aiStockSelect.value) aiStockSelect.value = stockSelect.value;
        if (chartTypeSelect && aiChartType) aiChartType.value = chartTypeSelect.value;
        if (aiDateInput) aiDateInput.value = `${yyyy}-${mm}-${dd}`;
    }
    function closeAiModal() {
        if (!aiModal) return;
        aiModal.style.display = 'none';
        aiModal.setAttribute('aria-hidden', 'true');
    }
    if (aiBtn) aiBtn.addEventListener('click', openAiModal);
    if (aiClose) aiClose.addEventListener('click', closeAiModal);
    if (aiModalBackdrop) aiModalBackdrop.addEventListener('click', closeAiModal);

    // Helper to get stock info name/symbol for report
    async function getStockInfoQuick(symbol) {
        try {
            const payload = await fetchJson('/stock_info', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stock_symbol: symbol, stock: symbol })
            });
            return payload.info || {};
        } catch {
            return { name: symbol, symbol };
        }
    }

    // AI submit logic
    if (aiSubmit) aiSubmit.addEventListener('click', async () => {
        const stock = aiStockSelect.value || stockSelect.value;
        const date = aiDateInput.value || `${yyyy}-${mm}-${dd}`;
        const chartType = (aiChartType.value || 'bar');
        const viz = (aiVisualization.value || 'both');
        const modelKey = (aiModel.value || 'gemini');
        if (!stock) { alert('Please choose a stock.'); return; }
        if (!date) { alert('Please select a date.'); return; }

        try {
            const payload = await fetchJson('/predict', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stock_symbol: stock, prediction_date: date, chart_type: chartType, stock: stock, date: date })
            });

            // Determine selected model's prediction
            const keyToField = { gemini: 'lr_prediction', chatgpt: 'rf_prediction', deepseek: 'svr_prediction' };
            const predField = keyToField[modelKey];
            const selectedPred = payload[predField];
            const modelLabels = { gemini: 'Gemini', chatgpt: 'ChatGPT', deepseek: 'DeepSeek' };
            aiSelectedModelEl.textContent = modelLabels[modelKey] || modelKey;
            aiPredictedPriceEl.textContent = (selectedPred != null) ? `$${Number(selectedPred).toFixed(2)}` : 'N/A';

            // Render images per visualization choice
            aiChartContainer.innerHTML = '';
            aiRankingContainer.innerHTML = '';
            if ((viz === 'both' || viz === 'chart') && payload.chart_image) {
                if (!aiChartImgEl) {
                    aiChartImgEl = document.createElement('img');
                    aiChartImgEl.style.width = '100%';
                    aiChartImgEl.style.maxHeight = '360px';
                    aiChartImgEl.style.objectFit = 'contain';
                    aiChartImgEl.alt = 'AI chart';
                    aiChartCaptionEl = document.createElement('div');
                    aiChartCaptionEl.style.marginTop = '6px';
                    aiChartCaptionEl.style.fontSize = '12px';
                    aiChartCaptionEl.style.color = '#5e6b7a';
                }
                aiChartImgEl.src = payload.chart_image;
                aiChartCaptionEl.textContent = `Model: ${modelLabels[modelKey] || modelKey} • Type: ${chartType.toUpperCase()}`;
                aiChartContainer.appendChild(aiChartImgEl);
                aiChartContainer.appendChild(aiChartCaptionEl);
            }
            if ((viz === 'both' || viz === 'ranking') && payload.ranking_image) {
                const rankImg = document.createElement('img');
                rankImg.src = payload.ranking_image;
                rankImg.alt = 'Model ranking';
                rankImg.style.width = '100%';
                rankImg.style.maxHeight = '260px';
                rankImg.style.objectFit = 'contain';
                const rankCaption = document.createElement('div');
                rankCaption.style.marginTop = '6px';
                rankCaption.style.fontSize = '12px';
                rankCaption.style.color = '#5e6b7a';
                rankCaption.textContent = `Highest: ${payload.highest_model || '-'} • Lowest: ${payload.lowest_model || '-'}`;
                aiRankingContainer.appendChild(rankImg);
                aiRankingContainer.appendChild(rankCaption);
            }

            // Prepare AI report data
            const info = await getStockInfoQuick(stock);
            aiLastPayloadForReport = {
                info: {
                    name: info.name || stock,
                    symbol: info.symbol || stock,
                    date: info.latest_date || date,
                    latestClose: typeof info.latest_close === 'number' ? info.latest_close : null,
                },
                selectedModel: modelLabels[modelKey] || modelKey,
                prediction: (selectedPred != null) ? Number(selectedPred) : null,
                chart_image: payload.chart_image,
                ranking_image: payload.ranking_image,
                chartType: chartType.toUpperCase(),
                visualization: viz
            };
            aiExportDocBtn.disabled = false;
        } catch (e) {
            console.error('AI prediction error:', e);
            alert(`Failed to run AI prediction: ${e.message}`);
        }
    });

    // AI export Word report
    if (aiExportDocBtn) aiExportDocBtn.addEventListener('click', () => {
        if (!aiLastPayloadForReport) { alert('Please run an AI prediction first.'); return; }
        const now = new Date();
        const title = `AI Stock Prediction - ${aiLastPayloadForReport.info.name || ''} (${aiLastPayloadForReport.info.symbol || ''}) - ${now.toISOString().slice(0,10)}`;
        const latest = typeof aiLastPayloadForReport.info.latestClose === 'number' ? aiLastPayloadForReport.info.latestClose : null;
        const pred = aiLastPayloadForReport.prediction;
        let changeCell = '-';
        if (latest != null && pred != null && !isNaN(pred)) {
            const diff = pred - latest;
            const dirUp = diff >= 0;
            changeCell = `${dirUp ? '▲' : '▼'} ${Math.abs(diff).toFixed(2)} (${((diff/latest)*100).toFixed(2)}%)`;
        }
        const tableRow = pred != null ? `<tr>
            <td style="padding:6px;border:1px solid #ccc;">${aiLastPayloadForReport.selectedModel}</td>
            <td style="padding:6px;border:1px solid #ccc;">${Number(pred).toFixed(2)}</td>
            <td style="padding:6px;border:1px solid #ccc;">${changeCell}</td>
        </tr>` : '';
        const docHtml = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${title}</title></head><body>
            <h2>${title}</h2>
            <p><strong>Report Generated:</strong> ${now.toLocaleDateString()} at ${now.toLocaleTimeString()}</p>
            <p><strong>Stock Information:</strong></p>
            <ul>
                <li><strong>Stock Name:</strong> ${aiLastPayloadForReport.info.name || 'N/A'}</li>
                <li><strong>Symbol:</strong> ${aiLastPayloadForReport.info.symbol || 'N/A'}</li>
                <li><strong>Current Price:</strong> ${latest != null ? '$' + latest.toFixed(2) : 'N/A'}</li>
                <li><strong>Analysis Date:</strong> ${aiLastPayloadForReport.info.date || 'N/A'}</li>
            </ul>

            <h3>AI Model Prediction</h3>
            <p><strong>Selected Model:</strong> ${aiLastPayloadForReport.selectedModel}</p>
            <table style="border-collapse:collapse;border:1px solid #ccc;width:100%;margin:20px 0;">
                <thead><tr style="background-color:#f8f9fa;">
                    <th style="padding:12px;border:1px solid #ccc;text-align:left;">Model</th>
                    <th style="padding:12px;border:1px solid #ccc;text-align:center;">Predicted Price</th>
                    <th style="padding:12px;border:1px solid #ccc;text-align:center;">Change vs Today</th>
                </tr></thead>
                <tbody>${tableRow}</tbody>
            </table>

            <h3>Visualization</h3>
            <p><strong>Chart Type:</strong> ${aiLastPayloadForReport.chartType} • <strong>View:</strong> ${aiLastPayloadForReport.visualization.toUpperCase()}</p>

            <p><strong>Disclaimer:</strong> Predictions are based on historical data and machine learning models. This is not financial advice.</p>
        </body></html>`;
        const blob = new Blob([`\ufeff${docHtml}`], { type: 'application/msword' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `${title}.doc`;
        a.click();
        URL.revokeObjectURL(url);
    });
    stockSelect.addEventListener('change', async () => {
        controlsCard.classList.add('slide-left');
        document.body.classList.add('split');
        setTimeout(() => controlsCard.classList.remove('slide-left'), 350);

        try {
            const payload = await fetchJson('/stock_info', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stock_symbol: stockSelect.value, stock: stockSelect.value })
            });

            const info = payload.info || {};
            stockNameEl.textContent = info.name || stockSelect.value;
            sumSymbol.textContent = info.symbol || '-';
            sumClose.textContent = info.latest_close != null ? Number(info.latest_close).toFixed(2) : '-';
            sumDate.textContent = info.latest_date || '-';
            sumHigh.textContent = info.day_high != null ? Number(info.day_high).toFixed(2) : '-';
            sumLow.textContent = info.day_low != null ? Number(info.day_low).toFixed(2) : '-';
            sumVolume.textContent = info.volume != null ? info.volume.toLocaleString() : '-';

            // KPIs
            kpiPrice.textContent = info.latest_close != null ? `$${Number(info.latest_close).toFixed(2)}` : '-';
            if (typeof info.change_pct === 'number') {
                const isUp = info.change_pct >= 0;
                kpiChange.textContent = `${isUp ? '▲' : '▼'} ${Math.abs(info.change_pct).toFixed(2)}%`;
                kpiChange.style.color = isUp ? '#1aa053' : '#d83a3a';
            } else {
                kpiChange.textContent = '-';
                kpiChange.style.color = '#0d2b6e';
            }
            kpiVolume.textContent = info.volume != null ? info.volume.toLocaleString() : '-';

            // Sparkline placeholder text; could be replaced with tiny inline SVG later
            if (Array.isArray(payload.historical_prices) && payload.historical_prices.length) {
                sparkline.textContent = 'Recent trend updated';
            } else {
                sparkline.textContent = 'No recent data';
            }
        } catch (e) {
            console.error('Error loading stock info:', e);
            alert(`Failed to load stock info: ${e.message}`);
        }
    });

    predictButton.addEventListener('click', async () => {
        const stock = stockSelect.value; const date = dateInput.value; const chartType = chartTypeSelect.value;
        if (!date) { alert('Please select a date.'); return; }

        try {
            const payload = await fetchJson('/predict', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stock_symbol: stock, prediction_date: date, chart_type: chartType, stock: stock, date: date })
            });

            const rows = [];
            if (payload.lr_prediction != null) rows.push({ name: 'Linear Regression', value: payload.lr_prediction });
            if (payload.rf_prediction != null) rows.push({ name: 'Random Forest', value: payload.rf_prediction });
            if (payload.svr_prediction != null) rows.push({ name: 'Support Vector Regression', value: payload.svr_prediction });
            if (payload.lstm_prediction != null) rows.push({ name: 'LSTM', value: payload.lstm_prediction });
            if (payload.arima_prediction != null) rows.push({ name: 'ARIMA', value: payload.arima_prediction });
            predictionsBody.innerHTML = rows.map(r => `<tr><td>${r.name}</td><td>${Number(r.value).toFixed(2)}</td></tr>`).join('');

            // Render backend matplotlib image and caption
            if (payload.chart_image) {
                if (!chartImgEl) {
                    chartImgEl = document.createElement('img');
                    chartImgEl.style.width = '100%'; chartImgEl.style.maxHeight = '360px'; chartImgEl.style.objectFit = 'contain'; chartImgEl.alt = 'Stock chart';
                    chartCaptionEl = document.createElement('div');
                    chartCaptionEl.style.marginTop = '6px'; chartCaptionEl.style.fontSize = '12px'; chartCaptionEl.style.color = '#5e6b7a';
                    chartContainer.appendChild(chartImgEl); chartContainer.appendChild(chartCaptionEl);
                }
                chartImgEl.src = payload.chart_image;
                chartCaptionEl.textContent = `Models: ALL • Type: ${chartType.toUpperCase()}`;
            }

            // Render ranking image and summary
            rankingContainer.innerHTML = '';
            if (payload.ranking_image) {
                const rankImg = document.createElement('img');
                rankImg.src = payload.ranking_image;
                rankImg.alt = 'Model ranking';
                rankImg.style.width = '100%';
                rankImg.style.maxHeight = '260px';
                rankImg.style.objectFit = 'contain';
                const rankCaption = document.createElement('div');
                rankCaption.style.marginTop = '6px'; rankCaption.style.fontSize = '12px'; rankCaption.style.color = '#5e6b7a';
                rankCaption.textContent = `Highest: ${payload.highest_model || '-'} • Lowest: ${payload.lowest_model || '-'}`;
                rankingContainer.appendChild(rankImg);
                rankingContainer.appendChild(rankCaption);
            }

            // Save minimal data for comparison page
            const comparePayload = {
                highest_model: payload.highest_model,
                lowest_model: payload.lowest_model,
                ranking_image: payload.ranking_image,
                pie_ranking_image: payload.pie_ranking_image
            };
            sessionStorage.setItem('lastCompare', JSON.stringify(comparePayload));

            // Stash for report export
            lastPayloadForReport = {
                info: { name: stockNameEl.textContent, symbol: sumSymbol.textContent, date: sumDate.textContent, latestClose: Number(sumClose.textContent) || null },
                rows,
                chart_image: payload.chart_image,
                ranking_image: payload.ranking_image
            };

            // Clear any previous visualization data
            sessionStorage.removeItem('visualizationData');
        } catch (error) {
            console.error('Error fetching prediction:', error);
            alert(`Failed to fetch prediction: ${error.message}`);
        }
    });

    compareButton.addEventListener('click', () => {
        window.open('compare.html', '_blank');
    });

    // Export Word report (.doc via HTML)
    exportDocBtn.addEventListener('click', () => {
        if (!lastPayloadForReport) { alert('Please run a prediction first.'); return; }
        const now = new Date();
        const title = `Stock Prediction Report - ${lastPayloadForReport.info.name || ''} (${lastPayloadForReport.info.symbol || ''}) - ${now.toISOString().slice(0,10)}`;
        const latest = typeof lastPayloadForReport.info.latestClose === 'number' ? lastPayloadForReport.info.latestClose : null;
        const tableRows = lastPayloadForReport.rows.map(r => {
            const pred = Number(r.value);
            let changeCell = '-';
            if (latest != null && !isNaN(pred)) {
                const diff = pred - latest;
                const dirUp = diff >= 0;
                changeCell = `${dirUp ? '▲' : '▼'} ${Math.abs(diff).toFixed(2)} (${((diff/latest)*100).toFixed(2)}%)`;
            }
            return `<tr>
                <td style="padding:6px;border:1px solid #ccc;">${r.name}</td>
                <td style="padding:6px;border:1px solid #ccc;">${pred.toFixed(2)}</td>
                <td style="padding:6px;border:1px solid #ccc;">${changeCell}</td>
            </tr>`;
        }).join('');
        const docHtml = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${title}</title></head><body>
            <h2>${title}</h2>
            <p><strong>Report Generated:</strong> ${now.toLocaleDateString()} at ${now.toLocaleTimeString()}</p>
            <p><strong>Stock Information:</strong></p>
            <ul>
                <li><strong>Stock Name:</strong> ${lastPayloadForReport.info.name || 'N/A'}</li>
                <li><strong>Symbol:</strong> ${lastPayloadForReport.info.symbol || 'N/A'}</li>
                <li><strong>Current Price:</strong> ${lastPayloadForReport.info.latestClose ? '$' + lastPayloadForReport.info.latestClose.toFixed(2) : 'N/A'}</li>
                <li><strong>Analysis Date:</strong> ${lastPayloadForReport.info.date || 'N/A'}</li>
            </ul>
            
            <h3>ML Model Predictions</h3>
            <p>The following predictions were generated using advanced machine learning algorithms including Linear Regression, Random Forest, Support Vector Regression, LSTM Neural Networks, and ARIMA time series analysis.</p>
            <table style="border-collapse:collapse;border:1px solid #ccc;width:100%;margin:20px 0;">
                <thead><tr style="background-color:#f8f9fa;">
                    <th style="padding:12px;border:1px solid #ccc;text-align:left;">Model</th>
                    <th style="padding:12px;border:1px solid #ccc;text-align:center;">Predicted Price</th>
                    <th style="padding:12px;border:1px solid #ccc;text-align:center;">Change vs Today</th>
                </tr></thead>
                <tbody>${tableRows}</tbody>
            </table>
            
            <h3>Analysis Summary</h3>
            <p><strong>Chart Type:</strong> The analysis includes multiple visualization types including histograms, line charts, area charts, box plots, and violin plots to provide comprehensive insights into stock price distributions and trends.</p>
            
            <p><strong>Model Performance:</strong> Each machine learning model has been trained on historical data and evaluated using performance metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared scores to ensure prediction accuracy.</p>
            
            <p><strong>Data Source:</strong> Historical stock data was sourced from Yahoo Finance (yfinance) covering the most recent 3-month period to capture current market trends and volatility patterns.</p>
            
            <p><strong>Disclaimer:</strong> These predictions are based on historical data and machine learning algorithms. Stock market predictions are inherently uncertain and should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.</p>
            
            <hr style="margin:30px 0;border:1px solid #eee;">
            <p style="color:#666;font-size:12px;">Report generated by Stock Price Prediction System - Advanced ML Analytics Platform</p>
        </body></html>`;
        const blob = new Blob([`\ufeff${docHtml}`], { type: 'application/msword' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `${title}.doc`;
        a.click();
        URL.revokeObjectURL(url);
    });


});