document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const backButton = document.getElementById('back-button');
    const themeToggle = document.getElementById('theme-toggle');
    const pieChartBtn = document.getElementById('pie-chart-btn');
    const barChartBtn = document.getElementById('bar-chart-btn');
    const chartContainer = document.getElementById('chart-container');
    const goToDashboardBtn = document.getElementById('go-to-dashboard');

    // Theme management
    const isDark = localStorage.getItem('theme') === 'dark';
    if (isDark) {
        document.body.classList.add('dark');
        themeToggle.textContent = 'Light Theme';
    }

    // Event listeners
    backButton.addEventListener('click', () => {
        window.location.href = 'dashboard.html';
    });

    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark');
        const isDark = document.body.classList.contains('dark');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        themeToggle.textContent = isDark ? 'Light Theme' : 'Dark Theme';
        
        // Re-render chart if data exists
        if (window.lastVisualizationData) {
            if (pieChartBtn.classList.contains('active')) {
                createPieChart();
            } else {
                createBarChart();
            }
        }
    });

    goToDashboardBtn.addEventListener('click', () => {
        window.location.href = 'dashboard.html';
    });

    // Chart type buttons
    pieChartBtn.addEventListener('click', () => {
        pieChartBtn.classList.add('active');
        barChartBtn.classList.remove('active');
        createPieChart();
    });

    barChartBtn.addEventListener('click', () => {
        barChartBtn.classList.add('active');
        pieChartBtn.classList.remove('active');
        createBarChart();
    });

    // Load data from session storage
    loadVisualizationData();

    // Handle window resize for better chart alignment
    window.addEventListener('resize', () => {
        if (window.lastVisualizationData) {
            if (pieChartBtn.classList.contains('active')) {
                createPieChart();
            } else {
                createBarChart();
            }
        }
    });

    // Function to load data from session storage
    function loadVisualizationData() {
        const data = sessionStorage.getItem('visualizationData');
        if (data) {
            try {
                window.lastVisualizationData = JSON.parse(data);
                createPieChart(); // Show default pie chart
                hidePlaceholder();
            } catch (error) {
                console.error('Error parsing visualization data:', error);
                showPlaceholder();
            }
        } else {
            showPlaceholder();
        }
    }

    // Function to hide placeholder
    function hidePlaceholder() {
        const placeholder = document.querySelector('.placeholder-text');
        if (placeholder) {
            placeholder.style.display = 'none';
        }
    }

    // Function to show placeholder
    function showPlaceholder() {
        const placeholder = document.querySelector('.placeholder-text');
        if (placeholder) {
            placeholder.style.display = 'block';
        }
    }

    // Function to create pie chart
    function createPieChart() {
        if (!window.lastVisualizationData) {
            showPlaceholder();
            return;
        }

        const data = window.lastVisualizationData.rows.map(row => ({
            name: row.name,
            value: parseFloat(row.value)
        }));

        const trace = {
            values: data.map(d => d.value),
            labels: data.map(d => d.name),
            type: 'pie',
            hole: 0.4,
            marker: {
                colors: ['#8b5cf6', '#ef4444', '#10b981', '#f59e0b', '#3b82f6'],
                line: {
                    color: '#ffffff',
                    width: 2
                }
            },
            textinfo: 'label+percent',
            textposition: 'outside',
            hoverinfo: 'label+value+percent',
            textfont: {
                size: 14,
                color: document.body.classList.contains('dark') ? '#e0e0e0' : '#0b1a2b'
            }
        };

        const layout = {
            title: {
                text: 'Model Predictions Distribution',
                font: {
                    size: 20,
                    color: document.body.classList.contains('dark') ? '#e0e0e0' : '#0b1a2b'
                },
                x: 0.5,
                y: 0.95
            },
            showlegend: true,
            legend: {
                x: 0.5,
                y: -0.1,
                orientation: 'h',
                font: {
                    size: 12,
                    color: document.body.classList.contains('dark') ? '#e0e0e0' : '#0b1a2b'
                }
            },
            margin: { t: 80, b: 80, l: 50, r: 50 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            height: 500,
            width: 800,
            autosize: true
        };

        Plotly.newPlot(chartContainer, [trace], layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false,
            useResizeHandler: true
        });
    }

    // Function to create bar chart
    function createBarChart() {
        if (!window.lastVisualizationData) {
            showPlaceholder();
            return;
        }

        const data = window.lastVisualizationData.rows.map(row => ({
            name: row.name,
            value: parseFloat(row.value)
        }));

        const trace = {
            x: data.map(d => d.name),
            y: data.map(d => d.value),
            type: 'bar',
            marker: {
                color: data.map((d, i) => ['#8b5cf6', '#ef4444', '#10b981', '#f59e0b', '#3b82f6'][i % 5]),
                line: {
                    color: '#ffffff',
                    width: 2
                }
            },
            text: data.map(d => '$' + d.value.toFixed(2)),
            textposition: 'auto',
            hoverinfo: 'x+y',
            textfont: {
                size: 12,
                color: document.body.classList.contains('dark') ? '#e0e0e0' : '#0b1a2b'
            }
        };

        const layout = {
            title: {
                text: 'Model Predictions Comparison',
                font: {
                    size: 20,
                    color: document.body.classList.contains('dark') ? '#e0e0e0' : '#0b1a2b'
                },
                x: 0.5,
                y: 0.95
            },
            xaxis: {
                title: {
                    text: 'ML Models',
                    font: {
                        size: 14,
                        color: document.body.classList.contains('dark') ? '#e0e0e0' : '#0b1a2b'
                    }
                },
                tickangle: -45,
                tickfont: {
                    size: 12,
                    color: document.body.classList.contains('dark') ? '#e0e0e0' : '#0b1a2b'
                },
                gridcolor: document.body.classList.contains('dark') ? '#404040' : '#e5e7eb'
            },
            yaxis: {
                title: {
                    text: 'Predicted Price ($)',
                    font: {
                        size: 14,
                        color: document.body.classList.contains('dark') ? '#e0e0e0' : '#0b1a2b'
                    }
                },
                gridcolor: document.body.classList.contains('dark') ? '#404040' : '#e5e7eb',
                tickfont: {
                    color: document.body.classList.contains('dark') ? '#e0e0e0' : '#0b1a2b'
                }
            },
            margin: { t: 80, b: 100, l: 80, r: 50 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            bargap: 0.3,
            height: 500,
            width: 800,
            autosize: true
        };

        Plotly.newPlot(chartContainer, [trace], layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false,
            useResizeHandler: true
        });
    }
});
