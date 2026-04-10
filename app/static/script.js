// ========================================
// SeismoSense - JavaScript
// ========================================

// Tab Navigation
const tabs = document.querySelectorAll('.nav-tab');
const panels = document.querySelectorAll('.tab-panel');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and panels
        tabs.forEach(t => t.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding panel
        tab.classList.add('active');
        const panelId = tab.dataset.tab + '-panel';
        document.getElementById(panelId).classList.add('active');
    });
});

// Toast Notification Functions
function getToastIcon(type) {
    const icons = {
        success: '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        error: '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
        warning: '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'
    };
    return icons[type] || icons.info;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            ${getToastIcon(type)}
        </svg>
        <span class="toast-message">${message}</span>
    `;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3000);
}

// Form Submission - Show loading state and toast
const form = document.getElementById('seismoForm');
const btn = document.getElementById('predictBtn');
const btnText = btn ? btn.querySelector('.btn-text') : null;
const loader = document.getElementById('loader');

if (form && btn && btnText) {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        btn.disabled = true;
        btnText.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Analyzing...';
        
        if (loader) {
            loader.style.display = 'inline-block';
        }

        const payload = {
            magnitude: parseFloat(document.getElementById('magnitude').value),
            depth: parseFloat(document.getElementById('depth').value),
            cdi: parseFloat(document.getElementById('cdi').value),
            mmi: parseFloat(document.getElementById('mmi').value),
            sig: parseFloat(document.getElementById('sig').value)
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showToast('Prediction successful!', 'success');
                const result = data.prediction;
                const confidences = data.probabilities;
                const confidence = confidences[result] * 100;
                
                const resultContainer = document.getElementById('result-container');
                if (resultContainer) {
                    resultContainer.innerHTML = `
                        <div class="result-card ${result}">
                            <div class="result-header">
                                <span class="result-label">Predicted Alert</span>
                                <span class="result-value">${result.toUpperCase()}</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill ${result}" id="confFill" style="width: 0%;"></div>
                            </div>
                            <p class="confidence-text">
                                Confidence: <span id="confValue">${confidence.toFixed(1)}</span>%
                            </p>
                        </div>
                    `;
                    
                    setTimeout(() => {
                        const fill = document.getElementById('confFill');
                        if (fill) fill.style.width = confidence.toFixed(1) + '%';
                    }, 100);
                }
            } else if (response.status === 422) {
                if (data.detail && Array.isArray(data.detail)) {
                    data.detail.forEach(err => {
                        const field = err.loc[err.loc.length - 1];
                        showToast(`Validation Error (${field}): ${err.msg}`, 'error');
                    });
                } else {
                    showToast('Validation Error', 'error');
                }
            } else {
                showToast(data.message || 'An error occurred', 'error');
            }
        } catch (error) {
            showToast('Network error or server down', 'error');
        } finally {
            btn.disabled = false;
            btnText.innerHTML = '<i class="fas fa-bolt"></i> Analyze & Predict';
            if (loader) {
                loader.style.display = 'none';
            }
        }
    });
}

// Clear button — reset all inputs
const clearBtn = document.getElementById('clearBtn');
if (clearBtn && form) {
    clearBtn.addEventListener('click', () => {
        form.querySelectorAll('.form-input').forEach(input => {
            input.value = '';
        });
        form.querySelector('.form-input').focus();
        
        const resultContainer = document.getElementById('result-container');
        if (resultContainer) {
            resultContainer.innerHTML = `
                <div class="empty-result">
                    <div class="empty-icon">
                        <i class="fas fa-seismic"></i>
                    </div>
                    <p class="empty-text">Enter earthquake parameters and click "Analyze &amp; Predict"</p>
                </div>
            `;
        }
    });
}

// File input display
const csvFileInput = document.getElementById('csvfile');
const fileNameDisplay = document.getElementById('fileName');
if (csvFileInput && fileNameDisplay) {
    csvFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameDisplay.textContent = e.target.files[0].name;
        }
    });
}

// Batch Prediction Form
const batchForm = document.getElementById('batchForm');
const batchBtn = document.getElementById('batchPredictBtn');
const batchLoader = document.getElementById('batchLoader');

if (batchForm && batchBtn) {
    batchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const fileInput = document.getElementById('csvfile');
        if (!fileInput || !fileInput.files.length) {
            showToast('Please select a CSV file', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('payload', fileInput.files[0]);

        const btnText = batchBtn.querySelector('.btn-text');
        batchBtn.disabled = true;
        if (btnText) {
            btnText.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Processing...';
        }
        if (batchLoader) {
            batchLoader.style.display = 'inline-block';
        }

        try {
            const response = await fetch('/predict/batch', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showToast('Batch prediction successful!', 'success');
                displayBatchResults(data);
            } else if (response.status === 422) {
                if (data.detail) {
                    showToast(data.detail, 'error');
                } else if (Array.isArray(data.detail)) {
                    data.detail.forEach(err => {
                        const field = err.loc ? err.loc[err.loc.length - 1] : 'validation';
                        showToast(`Validation Error (${field}): ${err.msg}`, 'error');
                    });
                } else {
                    showToast('Validation Error', 'error');
                }
            } else {
                showToast(data.message || 'An error occurred', 'error');
            }
        } catch (error) {
            showToast('Network error or server down', 'error');
        } finally {
            batchBtn.disabled = false;
            if (btnText) {
                btnText.innerHTML = '<i class="fas fa-bolt"></i> Run Batch Prediction';
            }
            if (batchLoader) {
                batchLoader.style.display = 'none';
            }
        }
    });
}

function displayBatchResults(data) {
    const container = document.getElementById('batch-result-container');
    if (!container) return;

    const predictions = data.prediction;
    const probabilities = data.probabilities;
    const count = predictions.length;

    const counts = { green: 0, orange: 0, red: 0, yellow: 0 };
    predictions.forEach(p => {
        if (counts[p] !== undefined) counts[p]++;
    });

    let tableRows = '';
    const predictionDetails = { green: [], orange: [], red: [], yellow: [] };
    predictions.forEach((pred, i) => {
        const probs = probabilities[i];
        const maxProb = Math.max(...probs) * 100;
        predictionDetails[pred].push(i + 1);
        tableRows += `
            <tr>
                <td>${i + 1}</td>
                <td class="alert-cell ${pred}">${pred.toUpperCase()}</td>
                <td>${maxProb.toFixed(1)}%</td>
            </tr>
        `;
    });

    container.innerHTML = `
        <div class="chart-container" style="display: block;">
            <canvas id="batchPieChart"></canvas>
        </div>
        <div class="batch-summary">
            <div class="batch-stat">
                <div class="batch-stat-value">${count}</div>
                <div class="batch-stat-label">Total Predictions</div>
            </div>
            <div class="batch-stat">
                <div class="batch-stat-value" style="color: var(--green);">${counts.green}</div>
                <div class="batch-stat-label">Green</div>
            </div>
            <div class="batch-stat">
                <div class="batch-stat-value" style="color: var(--orange);">${counts.orange}</div>
                <div class="batch-stat-label">Orange</div>
            </div>
            <div class="batch-stat">
                <div class="batch-stat-value" style="color: var(--red);">${counts.red}</div>
                <div class="batch-stat-label">Red</div>
            </div>
            <div class="batch-stat">
                <div class="batch-stat-value" style="color: var(--yellow);">${counts.yellow}</div>
                <div class="batch-stat-label">Yellow</div>
            </div>
        </div>
        <table class="batch-results-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Alert</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
                ${tableRows}
            </tbody>
        </table>
    `;

    const ctx = document.getElementById('batchPieChart');
    if (ctx) {
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Green', 'Orange', 'Red', 'Yellow'],
                datasets: [{
                    data: [counts.green, counts.orange, counts.red, counts.yellow],
                    backgroundColor: ['#22c55e', '#f97316', '#ef4444', '#eab308'],
                    borderColor: ['#22c55e', '#f97316', '#ef4444', '#eab308'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#8b949e'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const predictionList = predictionDetails[label.toLowerCase()];
                                return `${label}: ${value} predictions`;
                            },
                            afterLabel: function(context) {
                                const label = context.label || '';
                                const predictionList = predictionDetails[label.toLowerCase()];
                                if (predictionList && predictionList.length > 0) {
                                    return `Rows: ${predictionList.join(', ')}`;
                                }
                                return '';
                            }
                        }
                    }
                }
            }
        });
    }
}

// Clear batch button
const clearBatchBtn = document.getElementById('clearBatchBtn');
if (clearBatchBtn && batchForm) {
    clearBatchBtn.addEventListener('click', () => {
        const fileInput = document.getElementById('csvfile');
        if (fileInput) fileInput.value = '';
        
        const fileNameDisplay = document.getElementById('fileName');
        if (fileNameDisplay) fileNameDisplay.textContent = 'Choose CSV file...';
        
        const batchContainer = document.getElementById('batch-result-container');
        if (batchContainer) {
            batchContainer.innerHTML = `
                <div class="empty-result">
                    <div class="empty-icon">
                        <i class="fas fa-file-csv"></i>
                    </div>
                    <p class="empty-text">Upload a CSV file and click "Run Batch Prediction"</p>
                </div>
                <div class="chart-container" style="display: none;">
                    <canvas id="batchPieChart"></canvas>
                </div>
            `;
        }
    });
}
