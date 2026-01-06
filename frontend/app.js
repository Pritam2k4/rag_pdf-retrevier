/**
 * Memoria - Personal Knowledge Q&A
 * Frontend Application Logic
 */

// ═══════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════

const API_BASE_URL = 'http://localhost:8000';

// ═══════════════════════════════════════════════════════════════
// State Management
// ═══════════════════════════════════════════════════════════════

const state = {
    documents: [],
    conversations: [],
    isLoading: false,
    sidebarOpen: window.innerWidth > 768,
    theme: localStorage.getItem('theme') || 'light'
};

// ═══════════════════════════════════════════════════════════════
// DOM Elements
// ═══════════════════════════════════════════════════════════════

const elements = {
    // Sidebar
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.getElementById('sidebarToggle'),
    mobileMenuBtn: document.getElementById('mobileMenuBtn'),
    documentsList: document.getElementById('documentsList'),
    docCount: document.getElementById('docCount'),
    uploadZone: document.getElementById('uploadZone'),
    uploadBtn: document.getElementById('uploadBtn'),
    fileInput: document.getElementById('fileInput'),
    historyList: document.getElementById('historyList'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),

    // Main content
    chatContainer: document.getElementById('chatContainer'),
    welcomeScreen: document.getElementById('welcomeScreen'),
    messages: document.getElementById('messages'),
    questionInput: document.getElementById('questionInput'),
    sendBtn: document.getElementById('sendBtn'),

    // Theme
    themeToggle: document.getElementById('themeToggle'),

    // Loading & Toast
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    toastContainer: document.getElementById('toastContainer')
};

// ═══════════════════════════════════════════════════════════════
// API Functions
// ═══════════════════════════════════════════════════════════════

const api = {
    async uploadDocument(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        return response.json();
    },

    async query(question, k = 4) {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, k })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }

        return response.json();
    },

    async getDocuments() {
        const response = await fetch(`${API_BASE_URL}/documents`);
        if (!response.ok) throw new Error('Failed to fetch documents');
        return response.json();
    },

    async deleteDocument(docId) {
        const response = await fetch(`${API_BASE_URL}/documents/${docId}`, {
            method: 'DELETE'
        });
        if (!response.ok) throw new Error('Failed to delete document');
        return response.json();
    },

    async getConversations() {
        const response = await fetch(`${API_BASE_URL}/conversations`);
        if (!response.ok) throw new Error('Failed to fetch conversations');
        return response.json();
    },

    async clearConversations() {
        const response = await fetch(`${API_BASE_URL}/conversations`, {
            method: 'DELETE'
        });
        if (!response.ok) throw new Error('Failed to clear conversations');
        return response.json();
    }
};

// ═══════════════════════════════════════════════════════════════
// UI Functions
// ═══════════════════════════════════════════════════════════════

function showLoading(text = 'Processing...') {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.hidden = false;
    state.isLoading = true;
}

function hideLoading() {
    elements.loadingOverlay.hidden = true;
    state.isLoading = false;
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-message">${message}</span>
        <button class="toast-close" aria-label="Close">×</button>
    `;

    elements.toastContainer.appendChild(toast);

    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('click', () => toast.remove());

    // Auto-remove after 5 seconds
    setTimeout(() => toast.remove(), 5000);
}

function toggleSidebar() {
    state.sidebarOpen = !state.sidebarOpen;
    if (window.innerWidth <= 768) {
        elements.sidebar.classList.toggle('open', state.sidebarOpen);
    } else {
        elements.sidebar.classList.toggle('collapsed', !state.sidebarOpen);
    }
}

function setTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
}

function toggleTheme() {
    setTheme(state.theme === 'light' ? 'dark' : 'light');
}

// ═══════════════════════════════════════════════════════════════
// Document Management
// ═══════════════════════════════════════════════════════════════

function renderDocuments() {
    if (state.documents.length === 0) {
        elements.documentsList.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📄</span>
                <span>No documents yet</span>
            </div>
        `;
    } else {
        elements.documentsList.innerHTML = state.documents.map(doc => `
            <div class="document-item" data-id="${doc.id}">
                <span class="doc-icon">📄</span>
                <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
                <button class="doc-delete" data-id="${doc.id}" aria-label="Delete document">×</button>
            </div>
        `).join('');

        // Add delete handlers
        elements.documentsList.querySelectorAll('.doc-delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const docId = btn.dataset.id;
                await handleDeleteDocument(docId);
            });
        });
    }

    elements.docCount.textContent = state.documents.length;
}

async function handleFileUpload(files) {
    for (const file of files) {
        try {
            showLoading(`Uploading ${file.name}...`);
            const doc = await api.uploadDocument(file);
            state.documents.push(doc);
            showToast(`"${file.name}" uploaded successfully`, 'success');
        } catch (error) {
            showToast(`Failed to upload "${file.name}": ${error.message}`, 'error');
        }
    }
    hideLoading();
    renderDocuments();
}

async function handleDeleteDocument(docId) {
    try {
        await api.deleteDocument(docId);
        state.documents = state.documents.filter(d => d.id !== docId);
        renderDocuments();
        showToast('Document deleted', 'info');
    } catch (error) {
        showToast(`Failed to delete: ${error.message}`, 'error');
    }
}

async function loadDocuments() {
    try {
        state.documents = await api.getDocuments();
        renderDocuments();
    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

// ═══════════════════════════════════════════════════════════════
// Conversation Management
// ═══════════════════════════════════════════════════════════════

function renderHistory() {
    if (state.conversations.length === 0) {
        elements.historyList.innerHTML = `
            <div class="empty-state">
                <span>No conversations yet</span>
            </div>
        `;
    } else {
        // Group by date and show recent questions
        const recentQuestions = state.conversations.slice(-10).reverse();
        elements.historyList.innerHTML = recentQuestions.map((conv, i) => `
            <div class="history-item" data-index="${state.conversations.length - 1 - i}">
                ${conv.question.length > 40 ? conv.question.slice(0, 40) + '...' : conv.question}
            </div>
        `).join('');
    }
}

function addUserMessage(question) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-user';
    messageDiv.innerHTML = `
        <div class="message-content">${escapeHtml(question)}</div>
    `;
    elements.messages.appendChild(messageDiv);
    scrollToBottom();
}

function addTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message message-assistant';
    indicator.id = 'typingIndicator';
    indicator.innerHTML = `
        <div class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    elements.messages.appendChild(indicator);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
}

function addAssistantMessage(answer, sources) {
    removeTypingIndicator();

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-assistant';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="message-sources">
                <span class="sources-label">Sources</span>
                ${sources.map(source => `
                    <div class="source-item">
                        <span class="source-icon">📎</span>
                        <div class="source-info">
                            <span class="source-name">${escapeHtml(source.filename)} (chunk ${source.chunk}/${source.total_chunks})</span>
                            <span class="source-excerpt">${escapeHtml(source.excerpt)}</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-content">${formatAnswer(answer)}</div>
        ${sourcesHtml}
    `;

    elements.messages.appendChild(messageDiv);
    scrollToBottom();
}

function formatAnswer(text) {
    // Convert markdown-like formatting
    return escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>')
        .replace(/• /g, '&bull; ')
        .replace(/- /g, '&ndash; ');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

async function handleQuery() {
    const question = elements.questionInput.value.trim();
    if (!question || state.isLoading) return;

    // Hide welcome screen
    elements.welcomeScreen.classList.add('hidden');

    // Clear input and disable button
    elements.questionInput.value = '';
    elements.sendBtn.disabled = true;
    autoResize(elements.questionInput);

    // Add user message
    addUserMessage(question);

    // Show typing indicator
    addTypingIndicator();

    try {
        const result = await api.query(question);

        // Add to conversations
        state.conversations.push({
            question,
            answer: result.answer,
            sources: result.sources,
            timestamp: new Date().toISOString()
        });

        // Add assistant message
        addAssistantMessage(result.answer, result.sources);

        // Update history
        renderHistory();

    } catch (error) {
        removeTypingIndicator();
        showToast(`Error: ${error.message}`, 'error');
        addAssistantMessage('Sorry, I encountered an error processing your question. Please make sure the backend server is running and try again.', []);
    }
}

async function loadConversations() {
    try {
        state.conversations = await api.getConversations();
        renderHistory();

        // Render existing messages
        if (state.conversations.length > 0) {
            elements.welcomeScreen.classList.add('hidden');
            state.conversations.forEach(conv => {
                addUserMessage(conv.question);
                addAssistantMessage(conv.answer, conv.sources);
            });
        }
    } catch (error) {
        console.error('Failed to load conversations:', error);
    }
}

async function handleClearHistory() {
    if (!confirm('Clear all conversation history?')) return;

    try {
        await api.clearConversations();
        state.conversations = [];
        elements.messages.innerHTML = '';
        elements.welcomeScreen.classList.remove('hidden');
        renderHistory();
        showToast('History cleared', 'info');
    } catch (error) {
        showToast(`Failed to clear history: ${error.message}`, 'error');
    }
}

// ═══════════════════════════════════════════════════════════════
// Input Handling
// ═══════════════════════════════════════════════════════════════

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

function handleInputChange() {
    const hasText = elements.questionInput.value.trim().length > 0;
    elements.sendBtn.disabled = !hasText;
    autoResize(elements.questionInput);
}

function handleInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleQuery();
    }
}

// ═══════════════════════════════════════════════════════════════
// Drag and Drop
// ═══════════════════════════════════════════════════════════════

function setupDragAndDrop() {
    const zone = elements.uploadZone;

    ['dragenter', 'dragover'].forEach(event => {
        zone.addEventListener(event, (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        zone.addEventListener(event, (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
        });
    });

    zone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(Array.from(files));
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// Event Listeners
// ═══════════════════════════════════════════════════════════════

function setupEventListeners() {
    // Sidebar
    elements.sidebarToggle.addEventListener('click', toggleSidebar);
    elements.mobileMenuBtn.addEventListener('click', toggleSidebar);

    // Theme
    elements.themeToggle.addEventListener('click', toggleTheme);

    // File upload
    elements.uploadBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(Array.from(e.target.files));
            e.target.value = ''; // Reset input
        }
    });

    // History
    elements.clearHistoryBtn.addEventListener('click', handleClearHistory);

    // Input
    elements.questionInput.addEventListener('input', handleInputChange);
    elements.questionInput.addEventListener('keydown', handleInputKeydown);
    elements.sendBtn.addEventListener('click', handleQuery);

    // Responsive
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) {
            elements.sidebar.classList.remove('open');
        }
    });

    // Close sidebar on mobile when clicking outside
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 &&
            state.sidebarOpen &&
            !elements.sidebar.contains(e.target) &&
            !elements.mobileMenuBtn.contains(e.target)) {
            toggleSidebar();
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════

async function init() {
    // Set initial theme
    setTheme(state.theme);

    // Setup event listeners
    setupEventListeners();
    setupDragAndDrop();

    // Load initial data
    try {
        await Promise.all([
            loadDocuments(),
            loadConversations()
        ]);
    } catch (error) {
        console.error('Initialization error:', error);
        showToast('Unable to connect to server. Make sure the backend is running.', 'error');
    }

    // Focus input
    elements.questionInput.focus();
}

// Start the app
document.addEventListener('DOMContentLoaded', init);
