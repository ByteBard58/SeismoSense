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
