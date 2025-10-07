document.addEventListener('DOMContentLoaded', () => {
    const backButton = document.getElementById('back-button');
    const themeToggle = document.getElementById('theme-toggle');
    const stock1Select = document.getElementById('stock1');
    const stock2Select = document.getElementById('stock2');
    const stock3Select = document.getElementById('stock3');
    const compareDate = document.getElementById('compare-date');
    const compareButton = document.getElementById('compare-button');
    const resultsCard = document.getElementById('results-card');
    const comparisonSummary = document.getElementById('comparison-summary');
    const comparisonChart = document.getElementById('comparison-chart');
    const predictionsTable = document.getElementById('predictions-table');

    // Set default date to tomorrow
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    compareDate.value = tomorrow.toISOString().split('T')[0];

    // Theme toggle
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') document.body.classList.add('dark');
    themeToggle.textContent = savedTheme === 'dark' ? 'Light Theme' : 'Dark Theme';

    themeToggle.addEventListener('click', () => {
        const isDark = document.body.classList.toggle('dark');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        themeToggle.textContent = isDark ? 'Light Theme' : 'Dark Theme';
    });

    // Back button
    backButton.addEventListener('click', () => {
        window.location.href = 'dashboard.html';
    });

    // Compare button
    compareButton.addEventListener('click', async () => {
        const stock1 = stock1Select.value;
        const stock2 = stock2Select.value;
        const stock3 = stock3Select.value;
        const date = compareDate.value;

        if (!stock1 || !stock2 || !date) {
            alert('Please select at least 2 stocks and a prediction date.');
            return;
        }

        if (stock1 === stock2 || (stock3 && (stock1 === stock3 || stock2 === stock3))) {
            alert('Please select different stocks for comparison.');
            return;
        }

        compareButton.disabled = true;
        compareButton.textContent = 'Comparing...';

        try {
            const stocks = [stock1, stock2];
            if (stock3) stocks.push(stock3);

            const comparisonData = await compareStocks(stocks, date);
            displayComparisonResults(comparisonData);
            resultsCard.style.display = 'block';
        } catch (error) {
            console.error('Error comparing stocks:', error);
            alert('Error comparing stocks. Please try again.');
        } finally {
            compareButton.disabled = false;
            compareButton.textContent = 'Compare Stocks';
        }
    });

    async function compareStocks(stocks, date) {
        const results = [];
        
        for (const stock of stocks) {
            try {
                const response = await fetch(`http://127.0.0.1:5000/predict`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        stock_symbol: stock,
                        date: date,
                        chart_type: 'line',
                        chart_theme: document.body.classList.contains('dark') ? 'dark' : 'light'
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                results.push({
                    symbol: stock,
                    data: data,
                    predictions: data.predictions,
                    historical: data.historical_data,
                    model_metrics: data.model_metrics || {}
                });
            } catch (error) {
                console.error(`Error fetching data for ${stock}:`, error);
                results.push({
                    symbol: stock,
                    error: error.message
                });
            }
        }

        return results;
    }

    function displayComparisonResults(comparisonData) {
        // Display summary
        const validResults = comparisonData.filter(r => !r.error);
        comparisonSummary.innerHTML = `
            <div class="summary-grid">
                <div class="summary-item">
                    <h4>Stocks Compared</h4>
                    <p>${validResults.map(r => r.symbol).join(' vs ')}</p>
                </div>
                <div class="summary-item">
                    <h4>Prediction Date</h4>
                    <p>${compareDate.value}</p>
                </div>
                <div class="summary-item">
                    <h4>Models Used</h4>
                    <p>${validResults[0]?.predictions?.length || 0} ML Models</p>
                </div>
            </div>
        `;

        // Create interactive comparison chart
        createComparisonChart(validResults);

        // Display predictions table with model performance metrics
        createPredictionsTable(validResults);
    }

    function createComparisonChart(results) {
        if (!results.length) return;

        const traces = [];
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];

        results.forEach((result, index) => {
            if (result.historical && result.predictions) {
                // Historical data
                const historicalDates = result.historical.dates || [];
                const historicalPrices = result.historical.prices || [];
                
                traces.push({
                    x: historicalDates,
                    y: historicalPrices,
                    type: 'scatter',
                    mode: 'lines',
                    name: `${result.symbol} (Historical)`,
                    line: { color: colors[index], width: 2 },
                    opacity: 0.7
                });

                // Predictions
                const predictionDate = compareDate.value;
                const avgPrediction = result.predictions.reduce((sum, p) => sum + p.value, 0) / result.predictions.length;
                
                traces.push({
                    x: [historicalDates[historicalDates.length - 1], predictionDate],
                    y: [historicalPrices[historicalPrices.length - 1], avgPrediction],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: `${result.symbol} (Prediction)`,
                    line: { color: colors[index], width: 3, dash: 'dash' },
                    marker: { size: 8 }
                });
            }
        });

        const layout = {
            title: 'Stock Price Comparison & Predictions',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price ($)' },
            hovermode: 'closest',
            showlegend: true,
            legend: { x: 0, y: 1 },
            margin: { l: 50, r: 50, t: 80, b: 50 },
            plot_bgcolor: document.body.classList.contains('dark') ? '#2a2a4a' : '#f8fbff',
            paper_bgcolor: document.body.classList.contains('dark') ? '#1a1a2e' : '#eef3fb',
            font: {
                color: document.body.classList.contains('dark') ? '#e0e0e0' : '#0b1a2b'
            }
        };

        Plotly.newPlot(comparisonChart, traces, layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        });
    }

    function createPredictionsTable(results) {
        if (!results.length) return;

        const allPredictions = [];
        results.forEach(result => {
            if (result.predictions) {
                result.predictions.forEach(pred => {
                    allPredictions.push({
                        symbol: result.symbol,
                        model: pred.name,
                        predicted_price: pred.value,
                        metrics: result.model_metrics[pred.name] || {}
                    });
                });
            }
        });

        // Sort by predicted price
        allPredictions.sort((a, b) => b.predicted_price - a.predicted_price);

        const tableHTML = `
            <h4>Model Predictions & Performance Metrics</h4>
            <table class="predictions-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Stock</th>
                        <th>Model</th>
                        <th>Predicted Price</th>
                        <th>MSE</th>
                        <th>RMSE</th>
                        <th>R² Score</th>
                    </tr>
                </thead>
                <tbody>
                    ${allPredictions.map((pred, index) => `
                        <tr>
                            <td>${index + 1}</td>
                            <td><strong>${pred.symbol}</strong></td>
                            <td>${pred.model}</td>
                            <td>$${pred.predicted_price.toFixed(2)}</td>
                            <td>${pred.metrics.mse ? pred.metrics.mse.toFixed(4) : 'N/A'}</td>
                            <td>${pred.metrics.rmse ? pred.metrics.rmse.toFixed(4) : 'N/A'}</td>
                            <td>${pred.metrics.r2 ? pred.metrics.r2.toFixed(4) : 'N/A'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

        predictionsTable.innerHTML = tableHTML;
    }
});
