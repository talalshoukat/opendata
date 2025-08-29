// Chat Interface JavaScript
class ChatInterface {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.statusBadge = document.getElementById('statusBadge');
        this.chatHistory = [];
        
        this.initializeEventListeners();
        this.loadExampleQueries();
        this.checkHealth();
    }
    
    initializeEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => this.handleKeyPress(e));
    }
    
    handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage('user', message);
        
        // Clear input and disable send button
        this.messageInput.value = '';
        this.setLoadingState(true);
        
        try {
            // Show typing indicator
            this.showTypingIndicator();
            
            // Send message to API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            if (data.success) {
                console.log('Chat response data:', data);
                console.log('Has chart:', data.has_chart);
                console.log('Has report:', data.has_report);
                console.log('Has data:', data.has_data);
                console.log('Query ID:', data.query_id);
                
                // Add assistant response with action buttons if available
                this.addMessage('assistant', data.message, null, null, false, data.has_chart, data.has_report, data.has_data, data.query_id, data.data, data.columns, data.row_count, data.column_count);
            } else {
                // Show error message
                this.addMessage('assistant', `Sorry, I encountered an error: ${data.error || 'Unknown error'}`, null, null, true);
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('assistant', 'Sorry, I encountered a network error. Please try again.', null, null, true);
        } finally {
            this.hideTypingIndicator();
            this.setLoadingState(false);
        }
    }
    
    addMessage(role, content, chartData = null, vizCode = null, isError = false, hasChart = false, hasReport = false, hasData = false, queryId = null, dataframeData = null, columns = null, rowCount = null, columnCount = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${role}-message message-enter`;
        if (queryId) {
            messageDiv.setAttribute('data-query-id', queryId);
        }
        
        const timestamp = new Date().toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        let messageContent = `
            <div class="message-content ${isError ? 'error-message' : ''}">
                ${content}
            </div>
            <div class="message-time">${timestamp}</div>
        `;
        
        messageDiv.innerHTML = messageContent;
        
        // Add dataframe if available
        if (dataframeData && columns && rowCount && columnCount) {
            const dataframeContainer = this.createDataframeContainer(dataframeData, columns, rowCount, columnCount);
            messageDiv.appendChild(dataframeContainer);
        }
        
        // Add action buttons if available (below dataframe)
        if (hasChart || hasReport || hasData) {
            const actionButtons = this.createActionButtons(hasChart, hasReport, hasData, queryId);
            messageDiv.appendChild(actionButtons);
        }
        
        // Add chart if available
        if (chartData && vizCode) {
            const chartContainer = this.createChartContainer(chartData, vizCode);
            messageDiv.appendChild(chartContainer);
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Store in chat history
        this.chatHistory.push({
            role: role,
            content: content,
            timestamp: timestamp,
            chartData: chartData,
            vizCode: vizCode,
            hasChart: hasChart,
            hasReport: hasReport,
            queryId: queryId
        });
    }
    
    createActionButtons(hasChart, hasReport, hasData, queryId) {
        console.log('Creating action buttons:', { hasChart, hasReport, hasData, queryId });
        
        const actionDiv = document.createElement('div');
        actionDiv.className = 'action-buttons';
        
        let buttonsHTML = '<div class="action-buttons-container">';
        
        if (hasChart) {
            console.log('Adding Show Chart button');
            buttonsHTML += `
                <button class="btn btn-outline-primary btn-sm action-btn" onclick="generateChart('${queryId}')">
                    <i class="fas fa-chart-bar"></i> Show Chart
                </button>
            `;
        }
        
        if (hasReport) {
            console.log('Adding Generate Report button');
            buttonsHTML += `
                <button class="btn btn-outline-success btn-sm action-btn" onclick="generateReport('${queryId}')">
                    <i class="fas fa-file-pdf"></i> Generate Report
                </button>
            `;
        }
        
        buttonsHTML += '</div>';
        actionDiv.innerHTML = buttonsHTML;
        
        console.log('Action buttons HTML:', buttonsHTML);
        
        return actionDiv;
    }
    
    createDataframeContainer(data, columns, rowCount, columnCount) {
        console.log('Creating dataframe container with data:', data);
        console.log('Columns:', columns, 'Rows:', rowCount, 'Columns:', columnCount);
        
        const dataframeDiv = document.createElement('div');
        dataframeDiv.className = 'dataframe-container';
        
        // Create summary header
        const summaryHTML = `
            <div class="dataframe-summary">
                <h6><i class="fas fa-table"></i> Data Summary</h6>
                <p><strong>Rows:</strong> ${rowCount} | <strong>Columns:</strong> ${columnCount}</p>
            </div>
        `;
        
        // Create table
        let tableHTML = '<div class="table-responsive"><table class="table table-striped table-hover table-sm">';
        
        // Header row
        tableHTML += '<thead class="table-dark"><tr>';
        columns.forEach(col => {
            tableHTML += `<th scope="col">${col}</th>`;
        });
        tableHTML += '</tr></thead>';
        
        // Data rows (limit to first 50 rows for performance)
        tableHTML += '<tbody>';
        const displayData = data.slice(0, 50);
        displayData.forEach(row => {
            tableHTML += '<tr>';
            columns.forEach(col => {
                const value = row[col];
                const displayValue = value === null || value === undefined ? '' : String(value);
                tableHTML += `<td>${displayValue}</td>`;
            });
            tableHTML += '</tr>';
        });
        tableHTML += '</tbody></table></div>';
        
        // Add note if data is truncated
        if (data.length > 50) {
            tableHTML += `<div class="dataframe-note"><small class="text-muted">Showing first 50 rows of ${rowCount} total rows</small></div>`;
        }
        
        dataframeDiv.innerHTML = summaryHTML + tableHTML;
        
        return dataframeDiv;
    }
    
    createPNGChartContainer(base64Image) {
        console.log('Creating PNG chart container with base64 image');
        
        if (!base64Image) {
            console.error('No base64 image data provided');
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            chartDiv.innerHTML = '<div class="alert alert-danger">No chart image data available</div>';
            return chartDiv;
        }
        
        const chartDiv = document.createElement('div');
        chartDiv.className = 'chart-container';
        
        // Create image element with base64 data
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${base64Image}`;
        img.style.maxWidth = '100%';
        img.style.height = 'auto';
        img.style.border = '1px solid #ddd';
        img.style.borderRadius = '8px';
        img.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
        img.alt = 'Generated Chart';
        
        // Add loading state
        img.onload = () => {
            console.log('Chart image loaded successfully');
        };
        
        img.onerror = () => {
            console.error('Error loading chart image');
            img.alt = 'Error loading chart';
            img.style.border = '1px solid #dc3545';
        };
        
        chartDiv.appendChild(img);
        
        return chartDiv;
    }

    createChartContainer(plotData, plotLayout, plotConfig = {}) {
        console.log('Creating chart container with plot data:', plotData);
        console.log('Plot layout:', plotLayout);
        console.log('Plot config:', plotConfig);
        console.log('Plotly available:', typeof Plotly !== 'undefined');
        
        if (typeof Plotly === 'undefined') {
            console.error('Plotly is not loaded!');
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            chartDiv.innerHTML = '<div class="alert alert-danger">Plotly library not loaded. Please refresh the page.</div>';
            return chartDiv;
        }
        
        const chartDiv = document.createElement('div');
        chartDiv.className = 'chart-container';
        
        // Create a unique ID for the chart
        const chartId = `chart-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        chartDiv.innerHTML = `<div id="${chartId}"></div>`;
        
        // Use setTimeout to ensure DOM is ready before rendering
        setTimeout(() => {
            this.renderPlotlyChart(chartId, plotData, plotLayout, plotConfig);
        }, 100);
        
        return chartDiv;
    }
    
    renderPlotlyChart(chartId, plotData, plotLayout, plotConfig = {}) {
        console.log('Rendering Plotly chart with ID:', chartId);
        console.log('Plot data type:', typeof plotData, 'length:', plotData ? plotData.length : 'undefined');
        console.log('Plot data:', plotData);
        console.log('Plot layout type:', typeof plotLayout);
        console.log('Plot layout:', plotLayout);
        console.log('Plot config:', plotConfig);
        
        // Check if the element exists
        const chartElement = document.getElementById(chartId);
        if (!chartElement) {
            console.error('Chart element not found:', chartId);
            return;
        }
        
        try {
            if (!plotData || !plotLayout) {
                console.log('No plot data provided, showing error message');
                console.log('plotData is:', plotData);
                console.log('plotLayout is:', plotLayout);
                chartElement.innerHTML = '<div class="alert alert-warning">No chart data available</div>';
                return;
            }
            
            // Check if plotData is an array and has content
            if (!Array.isArray(plotData) || plotData.length === 0) {
                console.log('Plot data is not an array or is empty:', plotData);
                chartElement.innerHTML = '<div class="alert alert-warning">Plot data is empty or invalid</div>';
                return;
            }
            
            // Check the first trace
            const firstTrace = plotData[0];
            console.log('First trace:', firstTrace);
            console.log('First trace type:', firstTrace.type);
            console.log('First trace keys:', Object.keys(firstTrace));
            
            // Check for data in the first trace
            const hasData = firstTrace.x || firstTrace.y || firstTrace.values || firstTrace.labels;
            if (!hasData) {
                console.log('First trace has no data fields:', firstTrace);
                chartElement.innerHTML = '<div class="alert alert-warning">Chart trace has no data</div>';
                return;
            }
            
            // Merge default config with provided config
            const defaultConfig = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                displaylogo: false
            };
            const finalConfig = { ...defaultConfig, ...plotConfig };
            
            // Render the chart with Plotly
            console.log('Rendering chart with Plotly...');
            Plotly.newPlot(chartId, plotData, plotLayout, finalConfig);
            console.log('Chart rendered successfully');
            
        } catch (error) {
            console.error('Error rendering chart:', error);
            chartElement.innerHTML = '<div class="alert alert-danger">Error rendering chart: ' + error.message + '</div>';
        }
    }

    renderPreGeneratedChart(chartId, plotData, plotLayout) {
        // Legacy function - redirect to new function
        this.renderPlotlyChart(chartId, plotData, plotLayout);
    }
    
    renderChart(chartId, chartData, vizCode) {
        console.log('Rendering chart with ID:', chartId);
        
        // Check if the element exists
        const chartElement = document.getElementById(chartId);
        if (!chartElement) {
            console.error('Chart element not found:', chartId);
            return;
        }
        
        try {
            if (!vizCode || vizCode.trim() === '') {
                console.log('No visualization code provided, using fallback');
                this.createFallbackChart(chartId, chartData);
                return;
            }
            
            // Create a safe execution environment with necessary imports
            const executeVizCode = new Function('data', 'Plotly', 'pd', 'px', 'go', vizCode);
            
            // Mock pandas and plotly express for the execution
            const mockPd = {
                DataFrame: function(data) {
                    return {
                        to_dict: (orient) => data,
                        empty: data.length === 0,
                        shape: [data.length, Object.keys(data[0] || {}).length]
                    };
                }
            };
            
            const mockPx = {
                bar: (data, options) => {
                    console.log('Creating bar chart with data:', data, 'options:', options);
                    const traces = [];
                    const x = data.x || Object.keys(data[0] || {});
                    const y = data.y || Object.values(data[0] || {});
                    
                    traces.push({
                        x: x,
                        y: y,
                        type: 'bar',
                        name: options.title || 'Data'
                    });
                    
                    return {
                        data: traces,
                        layout: {
                            title: options.title || 'Chart',
                            xaxis: { title: options.x || 'X' },
                            yaxis: { title: options.y || 'Y' }
                        }
                    };
                },
                pie: (data, options) => {
                    console.log('Creating pie chart with data:', data, 'options:', options);
                    const values = data.values || Object.values(data[0] || {});
                    const labels = data.names || Object.keys(data[0] || {});
                    
                    return {
                        data: [{
                            values: values,
                            labels: labels,
                            type: 'pie'
                        }],
                        layout: {
                            title: options.title || 'Pie Chart'
                        }
                    };
                }
            };
            
            const mockGo = {
                Figure: function(data, layout) {
                    return { data: data, layout: layout };
                }
            };
            
            // Execute the code and get the figure
            console.log('Executing visualization code...');
            const fig = executeVizCode(chartData, Plotly, mockPd, mockPx, mockGo);
            console.log('Generated figure:', fig);
            
            // Render the chart
            if (fig && fig.data && fig.layout) {
                console.log('Rendering chart with Plotly...');
                Plotly.newPlot(chartId, fig.data, fig.layout, {responsive: true});
                console.log('Chart rendered successfully');
            } else {
                console.log('Figure data invalid, using fallback');
                // Fallback: create a simple bar chart from the data
                this.createFallbackChart(chartId, chartData);
            }
            
        } catch (error) {
            console.error('Chart rendering error:', error);
            console.log('Using fallback chart due to error');
            // Try fallback chart
            this.createFallbackChart(chartId, chartData);
        }
    }
    
    createFallbackChart(chartId, chartData) {
        console.log('Creating fallback chart with data:', chartData);
        
        // Check if the element exists
        const chartElement = document.getElementById(chartId);
        if (!chartElement) {
            console.error('Chart element not found for fallback:', chartId);
            return;
        }
        
        if (!chartData || chartData.length === 0) {
            console.log('No data available for fallback chart');
            chartElement.innerHTML = '<div class="alert alert-warning">No data available for chart</div>';
            return;
        }
        
        try {
            // Create a simple bar chart from the data
            const keys = Object.keys(chartData[0]);
            console.log('Data keys:', keys);
            
            if (keys.length < 2) {
                console.log('Not enough columns for chart');
                chartElement.innerHTML = '<div class="alert alert-warning">Not enough data columns for visualization</div>';
                return;
            }
            
            const xValues = chartData.map(row => row[keys[0]]);
            const yValues = chartData.map(row => {
                const val = row[keys[1]];
                return typeof val === 'number' ? val : 0;
            });
            
            console.log('X values:', xValues);
            console.log('Y values:', yValues);
            
            const trace = {
                x: xValues,
                y: yValues,
                type: 'bar',
                name: 'Data',
                marker: {
                    color: '#1f77b4'
                }
            };
            
            const layout = {
                title: 'Data Visualization',
                xaxis: { title: keys[0] },
                yaxis: { title: keys[1] || 'Count' },
                margin: { t: 50, r: 50, b: 50, l: 50 }
            };
            
            console.log('Rendering fallback chart...');
            Plotly.newPlot(chartId, [trace], layout, {responsive: true});
            console.log('Fallback chart rendered successfully');
            
        } catch (error) {
            console.error('Error creating fallback chart:', error);
            chartElement.innerHTML = '<div class="alert alert-danger">Error creating chart visualization</div>';
        }
    }
    
    showTypingIndicator() {
        this.typingIndicator.style.display = 'block';
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }
    
    setLoadingState(loading) {
        this.sendButton.disabled = loading;
        this.messageInput.disabled = loading;
        
        if (loading) {
            this.sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
            this.statusBadge.textContent = 'Processing';
            this.statusBadge.className = 'badge bg-warning';
        } else {
            this.sendButton.innerHTML = '<i class="fas fa-paper-plane"></i> Send';
            this.statusBadge.textContent = 'Ready';
            this.statusBadge.className = 'badge bg-success';
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    async loadExampleQueries() {
        try {
            const response = await fetch('/api/examples');
            const data = await response.json();
            
            const exampleContainer = document.getElementById('exampleQueries');
            exampleContainer.innerHTML = '';
            
            data.examples.forEach((example, index) => {
                const exampleDiv = document.createElement('div');
                exampleDiv.className = 'example-query';
                exampleDiv.textContent = example;
                exampleDiv.addEventListener('click', () => {
                    this.messageInput.value = example;
                    this.messageInput.focus();
                });
                exampleContainer.appendChild(exampleDiv);
            });
            
        } catch (error) {
            console.error('Error loading examples:', error);
        }
    }
    
    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (data.agent_initialized) {
                this.statusBadge.textContent = 'Ready';
                this.statusBadge.className = 'badge bg-success';
            } else {
                this.statusBadge.textContent = 'Initializing';
                this.statusBadge.className = 'badge bg-warning';
            }
        } catch (error) {
            this.statusBadge.textContent = 'Error';
            this.statusBadge.className = 'badge bg-danger';
        }
    }
    
    clearChat() {
        this.chatHistory = [];
        const messages = this.chatMessages.querySelectorAll('.user-message, .assistant-message');
        messages.forEach(message => message.remove());
        
        // Add welcome message back
        const welcomeMessage = document.createElement('div');
        welcomeMessage.className = 'welcome-message';
        welcomeMessage.innerHTML = `
            <div class="assistant-message">
                <div class="message-content">
                    <i class="fas fa-robot"></i> Hello! I'm your AI assistant. Ask me anything about your data in natural language.
                </div>
                <div class="message-time">${new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })}</div>
            </div>
        `;
        this.chatMessages.appendChild(welcomeMessage);
    }
    
    exportChat() {
        if (this.chatHistory.length === 0) {
            alert('No chat history to export');
            return;
        }
        
        const exportData = {
            exported_at: new Date().toISOString(),
            chat_history: this.chatHistory
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_history_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Global functions for HTML onclick handlers
function sendMessage() {
    window.chatInterface.sendMessage();
}

function handleKeyPress(event) {
    window.chatInterface.handleKeyPress(event);
}

function clearChat() {
    window.chatInterface.clearChat();
}

function exportChat() {
    window.chatInterface.exportChat();
}

async function generateChart(queryId) {
    try {
        console.log('Generating chart for query ID:', queryId);
        
        const response = await fetch(`/api/chart/${queryId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        console.log('Chart API response:', data);
        
        if (data.success) {
            console.log('Chart API returned success, processing response...');
            console.log('Chart data received:', data.chart_data);
            console.log('Plot data received:', data.plot_data);
            console.log('Plot layout received:', data.plot_layout);
            console.log('Plot config received:', data.plot_config);
            
            // Find the message with this query ID and add the chart
            const messageElement = document.querySelector(`[data-query-id="${queryId}"]`);
            console.log('Looking for message element with query ID:', queryId);
            console.log('Found message element:', messageElement);
            
            if (messageElement) {
                console.log('Found message element, creating chart container');
                try {
                    // Create chart container with Plotly data
                    const chartContainer = window.chatInterface.createChartContainer(data.plot_data, data.plot_layout, data.plot_config);
                    console.log('Chart container created with Plotly data:', chartContainer);
                    messageElement.appendChild(chartContainer);
                    console.log('Chart container appended to message element');
                    window.chatInterface.scrollToBottom();
                    console.log('Chart generation completed successfully');
                } catch (error) {
                    console.error('Error creating or appending chart container:', error);
                    alert('Error creating chart: ' + error.message);
                }
            } else {
                console.error('Message element not found for query ID:', queryId);
                console.log('Available elements with data-query-id:', document.querySelectorAll('[data-query-id]'));
                alert('Could not find the message to add chart to.');
            }
        } else {
            console.error('Chart generation failed:', data.error);
            alert(`Error generating chart: ${data.error}`);
        }
    } catch (error) {
        console.error('Error generating chart:', error);
        alert('Error generating chart. Please try again.');
    }
}



async function generateReport(queryId) {
    try {
        const response = await fetch(`/api/report/${queryId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response.ok) {
            // Get the filename from the Content-Disposition header
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = 'report.pdf';
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="(.+)"/);
                if (filenameMatch) {
                    filename = filenameMatch[1];
                }
            }
            
            // Create a blob from the response
            const blob = await response.blob();
            
            // Create a download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } else {
            const errorData = await response.json();
            alert(`Error generating report: ${errorData.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Error generating report:', error);
        alert('Error generating report. Please try again.');
    }
}

// Initialize chat interface when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Set the welcome message timestamp
    const welcomeTime = document.getElementById('welcomeTime');
    if (welcomeTime) {
        welcomeTime.textContent = new Date().toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }
    
    // Test Plotly availability
    console.log('Page loaded, checking Plotly availability...');
    console.log('Plotly available:', typeof Plotly !== 'undefined');
    if (typeof Plotly !== 'undefined') {
        console.log('Plotly version:', Plotly.version || 'unknown');
    }
    
    window.chatInterface = new ChatInterface();
});
