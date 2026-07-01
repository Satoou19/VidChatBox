// VidChatBox Frontend Application Logic

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Selectors & Config
    const providerSelect = document.getElementById("provider-select");
    const modelSelect = document.getElementById("model-select");
    const personaSelect = document.getElementById("persona-select");
    const activeProjectName = document.getElementById("active-project-name");
    
    // Collapsible Sidebar Elements
    const sidebar = document.getElementById("sidebar");
    const btnSidebarClose = document.getElementById("btn-sidebar-close");
    const btnSidebarOpen = document.getElementById("btn-sidebar-open");
    
    // Ingest VODs Modal Elements
    const ingestModal = document.getElementById("ingest-modal");
    const btnOpenIngest = document.getElementById("btn-open-ingest");
    const btnInputAttach = document.getElementById("btn-input-attach");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnCancelModal = document.getElementById("btn-cancel-modal");
    const ingestForm = document.getElementById("ingest-form");
    const vodUrlInput = document.getElementById("vod-urls");
    const btnIngest = document.getElementById("btn-ingest");
    const ingestSpinner = document.getElementById("ingest-spinner");
    
    // API Settings Modal Elements
    const settingsModal = document.getElementById("settings-modal");
    const btnOpenSettings = document.getElementById("btn-open-settings");
    const btnCloseSettings = document.getElementById("btn-close-settings");
    const btnCancelSettings = document.getElementById("btn-cancel-settings");
    const settingsForm = document.getElementById("settings-form");
    const userOpenRouterKey = document.getElementById("user-openrouter-key");
    const userGroqKey = document.getElementById("user-groq-key");
    const userOpenAIKey = document.getElementById("user-openai-key");
    const userGeminiKey = document.getElementById("user-gemini-key");
    const userDeepSeekKey = document.getElementById("user-deepseek-key");
    
    // Knowledge Base Elements
    const videoList = document.getElementById("video-list");
    const videoCount = document.getElementById("video-count");
    const emptyVideosMsg = document.getElementById("empty-videos-msg");
    const videoSearch = document.getElementById("video-search");
    
    // Chat Elements
    const chatMessages = document.getElementById("chat-messages");
    const welcomeContainer = document.getElementById("welcome-container");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const typingIndicator = document.getElementById("typing-indicator");
    const btnSend = document.getElementById("btn-send");

    // Group Manager Elements
    const groupList = document.getElementById("group-list");
    const btnAddGroupToggle = document.getElementById("btn-add-group-toggle");
    const inlineGroupForm = document.getElementById("inline-group-form");
    const inlineGroupName = document.getElementById("inline-group-name");
    const btnInlineGroupSave = document.getElementById("btn-inline-group-save");
    const btnInlineGroupCancel = document.getElementById("btn-inline-group-cancel");

    // Floating Progress Widget Elements
    const progressWidget = document.getElementById("progress-widget");
    const widgetOverallStatus = document.getElementById("widget-overall-status");
    const widgetProgressBar = document.getElementById("widget-progress-bar");
    const widgetProgressPercent = document.getElementById("widget-progress-percent");
    const widgetActiveTitle = document.getElementById("widget-active-title");
    const widgetTasksList = document.getElementById("widget-tasks-list");
    const btnWidgetToggle = document.getElementById("btn-widget-toggle");
    const btnWidgetClose = document.getElementById("btn-widget-close");

    // Base API URL (relative to server root)
    const API_BASE = "";

    // Active State
    let currentProjectId = "default";
    let activeChatHistory = [];

    // -------------------------------------------------------------
    // Initialization & Setup
    // -------------------------------------------------------------
    
    loadProjects();

    // Auto-resize chat textarea
    chatInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });

    // Enter key to submit, Shift+Enter to insert newline
    chatInput.addEventListener("keydown", function(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // AI Provider & Model Mapping (low tier high traffic preferred)
    const providerModels = {
        groq: [
            { value: "llama-3.1-8b-instant", text: "Llama 3.1 8B (Instant & Light)" },
            { value: "llama-3.3-70b-versatile", text: "Llama 3.3 70B (Versatile & Advanced)" }
        ],
        openrouter: [
            { value: "openai/gpt-4o-mini", text: "GPT-4o Mini (Cost-Effective)" },
            { value: "google/gemini-2.0-flash-001", text: "Gemini 2.0 Flash (Paid)" },
            { value: "anthropic/claude-3.5-sonnet", text: "Claude 3.5 Sonnet (Premium & Advanced)" },
            { value: "deepseek/deepseek-chat", text: "DeepSeek V3 (Reasoning & Code)" },
            { value: "meta-llama/llama-3.3-70b-instruct", text: "Llama 3.3 70B Instruct" }
        ],
        gemini: [
            { value: "gemini-2.0-flash", text: "Gemini 2.0 Flash" },
            { value: "gemini-1.5-flash", text: "Gemini 1.5 Flash" },
            { value: "gemini-1.5-pro", text: "Gemini 1.5 Pro" }
        ],
        openai: [
            { value: "gpt-4o-mini", text: "GPT-4o Mini" },
            { value: "gpt-4o", text: "GPT-4o" }
        ],
        deepseek: [
            { value: "deepseek-chat", text: "DeepSeek V3" }
        ]
    };

    function updateProviderDropdown() {
        const hasOpenAI = !!localStorage.getItem("user_openai_key");
        const hasGemini = !!localStorage.getItem("user_gemini_key");
        const hasDeepSeek = !!localStorage.getItem("user_deepseek_key");
        const hasOpenRouter = !!localStorage.getItem("user_openrouter_key");
        
        const optOpenAI = document.getElementById("opt-openai");
        const optGemini = document.getElementById("opt-gemini");
        const optDeepSeek = document.getElementById("opt-deepseek");
        const optOpenRouter = document.getElementById("opt-openrouter");
        
        if (optOpenAI) {
            if (hasOpenAI) {
                optOpenAI.disabled = false;
                optOpenAI.textContent = "OpenAI";
            } else {
                optOpenAI.disabled = true;
                optOpenAI.textContent = "OpenAI (Key required 🔒)";
            }
        }
        if (optGemini) {
            if (hasGemini) {
                optGemini.disabled = false;
                optGemini.textContent = "Gemini";
            } else {
                optGemini.disabled = true;
                optGemini.textContent = "Gemini (Key required 🔒)";
            }
        }
        if (optDeepSeek) {
            if (hasDeepSeek) {
                optDeepSeek.disabled = false;
                optDeepSeek.textContent = "DeepSeek";
            } else {
                optDeepSeek.disabled = true;
                optDeepSeek.textContent = "DeepSeek (Key required 🔒)";
            }
        }
        if (optOpenRouter) {
            if (hasOpenRouter) {
                optOpenRouter.disabled = false;
                optOpenRouter.textContent = "OpenRouter";
            } else {
                optOpenRouter.disabled = true;
                optOpenRouter.textContent = "OpenRouter (Key required 🔒)";
            }
        }
        
        // Prioritize auto-selecting active provider based on configured keys (openrouter > gemini > openai > deepseek > groq)
        if (hasOpenRouter) {
            providerSelect.value = "openrouter";
        } else if (hasGemini) {
            providerSelect.value = "gemini";
        } else if (hasOpenAI) {
            providerSelect.value = "openai";
        } else if (hasDeepSeek) {
            providerSelect.value = "deepseek";
        } else {
            providerSelect.value = "groq";
        }
    }

    function updateModelSelect() {
        const provider = providerSelect.value;
        const models = providerModels[provider] || [];
        modelSelect.innerHTML = "";
        models.forEach(model => {
            const opt = document.createElement("option");
            opt.value = model.value;
            opt.textContent = model.text;
            modelSelect.appendChild(opt);
        });
    }

    providerSelect.addEventListener("change", updateModelSelect);
    updateProviderDropdown();
    updateModelSelect();

    // -------------------------------------------------------------
    // Sidebar Collapse Logic
    // -------------------------------------------------------------
    
    btnSidebarClose.addEventListener("click", () => {
        sidebar.classList.add("collapsed");
        btnSidebarOpen.classList.remove("hidden");
    });

    btnSidebarOpen.addEventListener("click", () => {
        sidebar.classList.remove("collapsed");
        btnSidebarOpen.classList.add("hidden");
    });

    // -------------------------------------------------------------
    // Modal Overlay Logic
    // -------------------------------------------------------------
    
    const openIngestModal = () => {
        ingestModal.classList.remove("hidden");
        vodUrlInput.focus();
    };

    const closeIngestModal = () => {
        ingestModal.classList.add("hidden");
        vodUrlInput.value = "";
    };

    btnOpenIngest.addEventListener("click", openIngestModal);
    btnInputAttach.addEventListener("click", openIngestModal);
    btnCloseModal.addEventListener("click", closeIngestModal);
    btnCancelModal.addEventListener("click", closeIngestModal);
    
    ingestModal.addEventListener("click", (e) => {
        if (e.target === ingestModal) {
            closeIngestModal();
        }
    });

    // Update ingest method note dynamically
    const ingestProviderSelect = document.getElementById("ingest-provider-select");
    const ingestMethodNote = document.getElementById("ingest-method-note");
    if (ingestProviderSelect && ingestMethodNote) {
        ingestProviderSelect.addEventListener("change", () => {
            const val = ingestProviderSelect.value;
            if (val === "local-ai") {
                ingestMethodNote.innerHTML = `💻 <strong>Local AI (Sentence-Transformers):</strong> Sử dụng mô hình AI <code>intfloat/multilingual-e5-small</code> (~180MB) chạy hoàn toàn offline trên CPU. Tự động tải mô hình ở lần chạy đầu tiên. <strong>Tìm kiếm ngữ nghĩa (Semantic Search), hiểu từ đồng nghĩa, miễn phí và không tốn quota API!</strong>`;
                ingestMethodNote.style.borderLeftColor = "var(--text-success)";
                ingestMethodNote.style.background = "rgba(16, 185, 129, 0.06)";
            } else {
                const providerName = val.charAt(0).toUpperCase() + val.slice(1);
                ingestMethodNote.innerHTML = `🔮 <strong>AI Embedding (${providerName}):</strong> Phân tích ngữ nghĩa chiều sâu (semantic search) qua API của ${providerName}. Giúp tìm kiếm theo ý nghĩa và ngữ cảnh kể cả khi dùng từ đồng nghĩa, nhưng <strong>sẽ tiêu tốn quota API</strong> và yêu cầu điền API Key trong phần cài đặt.`;
                ingestMethodNote.style.borderLeftColor = "var(--accent-purple)";
                ingestMethodNote.style.background = "rgba(124, 58, 237, 0.06)";
            }
        });
    }

    // -------------------------------------------------------------
    // API Settings Modal Logic
    // -------------------------------------------------------------
    
    const openSettingsModal = () => {
        // Load saved API keys from localStorage
        userOpenRouterKey.value = localStorage.getItem("user_openrouter_key") || "";
        userGroqKey.value = localStorage.getItem("user_groq_key") || "";
        userOpenAIKey.value = localStorage.getItem("user_openai_key") || "";
        userGeminiKey.value = localStorage.getItem("user_gemini_key") || "";
        userDeepSeekKey.value = localStorage.getItem("user_deepseek_key") || "";
        
        settingsModal.classList.remove("hidden");
        userOpenRouterKey.focus();
    };

    const closeSettingsModal = () => {
        settingsModal.classList.add("hidden");
    };

    if (btnOpenSettings) btnOpenSettings.addEventListener("click", openSettingsModal);
    if (btnCloseSettings) btnCloseSettings.addEventListener("click", closeSettingsModal);
    if (btnCancelSettings) btnCancelSettings.addEventListener("click", closeSettingsModal);
    
    settingsModal.addEventListener("click", (e) => {
        if (e.target === settingsModal) {
            closeSettingsModal();
        }
    });

    settingsForm.addEventListener("submit", (e) => {
        e.preventDefault();
        
        // Save values to localStorage
        localStorage.setItem("user_openrouter_key", userOpenRouterKey.value.trim());
        localStorage.setItem("user_groq_key", userGroqKey.value.trim());
        localStorage.setItem("user_openai_key", userOpenAIKey.value.trim());
        localStorage.setItem("user_gemini_key", userGeminiKey.value.trim());
        localStorage.setItem("user_deepseek_key", userDeepSeekKey.value.trim());
        
        updateProviderDropdown();
        closeSettingsModal();
        showToast("API Settings saved successfully!", "success");
    });

    // -------------------------------------------------------------
    // Suggestion Cards Logic
    // -------------------------------------------------------------
    
    document.addEventListener("click", (e) => {
        const card = e.target.closest(".suggestion-card");
        if (card) {
            const prompt = card.getAttribute("data-prompt");
            if (prompt) {
                chatInput.value = prompt;
                chatInput.style.height = "auto";
                chatInput.style.height = (chatInput.scrollHeight) + "px";
                chatInput.focus();
                
                // Submit the form
                chatForm.dispatchEvent(new Event("submit"));
            }
        }
    });

    // -------------------------------------------------------------
    // VOD Ingestion Form Submission
    // -------------------------------------------------------------
    
    ingestForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const urlsText = vodUrlInput.value.trim();
        if (!urlsText) return;

        // Parse URLs (split by newline and filter empty lines)
        const urls = urlsText.split("\n").map(u => u.trim()).filter(u => u);
        if (urls.length === 0) return;

        // UI Feedback - Spinner and Disable
        btnIngest.disabled = true;
        ingestSpinner.classList.remove("hidden");

        // Prepare Progress Widget
        progressWidget.classList.remove("hidden");
        progressWidget.classList.remove("minimized");
        widgetOverallStatus.textContent = "Initializing ingestion...";
        widgetOverallStatus.style.color = "var(--text-primary)";
        widgetProgressPercent.textContent = "0%";
        widgetProgressBar.style.width = "0%";
        widgetActiveTitle.textContent = "Active: Initializing...";
        widgetTasksList.innerHTML = "";
        btnWidgetClose.classList.add("hidden");

        try {
            const provider = document.getElementById("ingest-provider-select").value;
            const openAiKey = localStorage.getItem("user_openai_key");
            const geminiKey = localStorage.getItem("user_gemini_key");
            const openRouterKey = localStorage.getItem("user_openrouter_key");
            
            if (urls.length === 1) {
                // Single Video Ingestion
                const response = await fetch(`${API_BASE}/api/ingest`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        url: urls[0], 
                        provider: provider, 
                        project_id: currentProjectId,
                        openai_key: openAiKey,
                        gemini_key: geminiKey,
                        openrouter_key: openRouterKey
                    })
                });

                if (!response.ok) {
                    throw new Error(await getErrorMessage(response, "Failed to start ingestion"));
                }

                const data = await response.json();
                closeIngestModal();
                
                // Start polling single ingestion status
                pollIngestionStatus(data.task_id);
            } else {
                // Batch Ingestion
                const response = await fetch(`${API_BASE}/api/ingest-batch`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        urls: urls, 
                        provider: provider, 
                        project_id: currentProjectId,
                        openai_key: openAiKey,
                        gemini_key: geminiKey,
                        openrouter_key: openRouterKey
                    })
                });

                if (!response.ok) {
                    throw new Error(await getErrorMessage(response, "Failed to start batch ingestion"));
                }

                const data = await response.json();
                closeIngestModal();
                
                // Start polling batch status
                pollBatchIngestionStatus(data.batch_task_id);
            }

        } catch (err) {
            showToast("Ingestion error: " + err.message, "error");
            widgetOverallStatus.textContent = "Error";
            widgetOverallStatus.style.color = "var(--text-error)";
            widgetActiveTitle.textContent = err.message;
        } finally {
            btnIngest.disabled = false;
            ingestSpinner.classList.add("hidden");
        }
    });

    // -------------------------------------------------------------
    // Real-Time Progress Floating Widget Logic
    // -------------------------------------------------------------

    // Expand/Minimize toggle
    btnWidgetToggle.addEventListener("click", (e) => {
        e.stopPropagation();
        progressWidget.classList.toggle("minimized");
    });

    progressWidget.querySelector(".widget-header").addEventListener("click", () => {
        progressWidget.classList.toggle("minimized");
    });

    // Close progress widget
    btnWidgetClose.addEventListener("click", (e) => {
        e.stopPropagation();
        progressWidget.classList.add("hidden");
    });

    // Helper: calculate step badge state class
    function getStepState(step, status, video) {
        if (status === "completed") {
            return "completed";
        }
        
        const steps = ["info", "download", "chunk", "index"];
        const stepIndex = steps.indexOf(step);
        
        let activeIndex = -1;
        if (status === "extracting_metadata") activeIndex = 0;
        else if (status === "downloading_subtitles" || status === "downloading") activeIndex = 1;
        else if (status === "processing_package" || status === "processing") activeIndex = 2;
        else if (status === "indexing") activeIndex = 3;
        
        if (status === "failed") {
            // Deduce where the pipeline crashed based on metadata
            let failedIndex = 0;
            if (video.title === video.url || !video.title) {
                failedIndex = 0; // Failed at extracting metadata
            } else {
                const err = (video.error || "").toLowerCase();
                if (err.includes("subtitle") || err.includes("download") || err.includes("yt-dlp") || err.includes("youtube")) {
                    failedIndex = 1; // Failed at subtitle fetch/download
                } else if (err.includes("chunk") || err.includes("segment") || err.includes("process")) {
                    failedIndex = 2; // Failed at transcript processing/chunking
                } else {
                    failedIndex = 3; // Failed at vector store indexing
                }
            }
            
            if (stepIndex < failedIndex) return "completed";
            if (stepIndex === failedIndex) return "failed";
            return "pending";
        }
        
        if (stepIndex < activeIndex) return "completed";
        if (stepIndex === activeIndex) return "active";
        return "pending";
    }

    // Unified UI renderer for Floating Ingestion Progress Widget
    function updateProgressWidget(data) {
        progressWidget.classList.remove("hidden");
        
        // Progress Summary
        const overallPercent = Math.round(data.overall_percent || 0);
        widgetProgressPercent.textContent = `${overallPercent}%`;
        widgetProgressBar.style.width = `${overallPercent}%`;
        
        // Header Text & Close button toggle
        if (data.status === "completed") {
            widgetOverallStatus.textContent = "Ingestion Complete!";
            widgetOverallStatus.style.color = "var(--text-success)";
            btnWidgetClose.classList.remove("hidden");
        } else if (data.status === "completed_with_errors") {
            widgetOverallStatus.textContent = "Completed with errors";
            widgetOverallStatus.style.color = "var(--text-error)";
            btnWidgetClose.classList.remove("hidden");
        } else if (data.status === "failed") {
            widgetOverallStatus.textContent = "Ingestion Failed";
            widgetOverallStatus.style.color = "var(--text-error)";
            btnWidgetClose.classList.remove("hidden");
        } else {
            widgetOverallStatus.textContent = `Ingesting VIDs (${data.completed_videos || 0}/${data.total_videos || 1})`;
            widgetOverallStatus.style.color = "var(--text-primary)";
            btnWidgetClose.classList.add("hidden");
        }
        
        // Active VID Subtitle text
        if (data.current_video_title) {
            widgetActiveTitle.textContent = `Active VID: ${data.current_video_title}`;
        } else {
            widgetActiveTitle.textContent = "Active: Initializing...";
        }
        
        // Render Checklist of Videos
        widgetTasksList.innerHTML = "";
        const videos = data.videos || [];
        
        videos.forEach(video => {
            const li = document.createElement("li");
            li.className = "widget-task-item";
            
            const statusDisplay = formatStatusName(video.status);
            const statusClass = (video.status === "completed") ? "completed" : 
                                (video.status === "failed") ? "failed" : "";
                                
            const infoState = getStepState("info", video.status, video);
            const downloadState = getStepState("download", video.status, video);
            const chunkState = getStepState("chunk", video.status, video);
            const indexState = getStepState("index", video.status, video);
            
            li.innerHTML = `
                <div class="widget-task-info">
                    <span class="widget-task-title" title="${escapeHtml(video.title || video.url)}">${escapeHtml(video.title || video.url)}</span>
                    <span class="widget-task-status ${statusClass}">${statusDisplay} ${video.percent ? Math.round(video.percent) + '%' : ''}</span>
                </div>
                <div class="widget-task-steps">
                    <span class="widget-step-badge ${infoState}">Info</span>
                    <span class="widget-step-badge ${downloadState}">Download</span>
                    <span class="widget-step-badge ${chunkState}">Chunk</span>
                    <span class="widget-step-badge ${indexState}">Index</span>
                </div>
                ${video.error ? `<div class="progress-video-title" style="color: var(--text-error); font-size: 0.7rem; white-space: normal; word-break: break-word; margin-top: 4px;">Error: ${escapeHtml(video.error)}</div>` : ''}
            `;
            widgetTasksList.appendChild(li);
        });
    }

    // Ingest status polling for single video
    function pollIngestionStatus(taskId) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`${API_BASE}/api/status/${taskId}`);
                if (!response.ok) throw new Error("Status lookup failed");

                const data = await response.json();
                
                // Update detailed floating widget UI
                updateProgressWidget(data);
                
                if (data.status === "completed") {
                    clearInterval(interval);
                    loadVideos(); // Refresh library
                    
                    // Add success message in chat
                    appendMessage("assistant", `Successfully ingested and indexed: **${data.title}**. You can now ask questions about it!`);
                } else if (data.status === "failed") {
                    clearInterval(interval);
                }

            } catch (err) {
                clearInterval(interval);
                console.error("Single task poll error: ", err);
            }
        }, 2000);
    }

    // Ingest status polling for batch of videos
    function pollBatchIngestionStatus(batchTaskId) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`${API_BASE}/api/batch-status/${batchTaskId}`);
                if (!response.ok) throw new Error("Batch status lookup failed");

                const data = await response.json();
                
                // Update detailed floating widget UI
                updateProgressWidget(data);

                if (data.status === "completed" || data.status === "completed_with_errors") {
                    clearInterval(interval);
                    loadVideos(); // Refresh library
                    
                    appendMessage("assistant", `Batch ingestion finished: **${data.completed_videos} out of ${data.total_videos}** videos successfully indexed!`);
                } else if (data.status === "failed") {
                    clearInterval(interval);
                }

            } catch (err) {
                clearInterval(interval);
                console.error("Batch task poll error: ", err);
            }
        }, 2000);
    }

    // Helper: format task status string for user view
    function formatStatusName(status) {
        switch(status) {
            case "extracting_metadata": return "Info Extraction...";
            case "downloading_subtitles": return "Subtitles Download...";
            case "downloading": return "Downloading...";
            case "transcribing": return "Transcribing...";
            case "processing_package": return "Chunking...";
            case "processing": return "Formatting...";
            case "indexing": return "Indexing Search DB...";
            case "completed": return "Complete";
            case "failed": return "Failed";
            case "pending": return "Pending";
            default: return status || "Processing...";
        }
    }

    // -------------------------------------------------------------
    // Chat Message Submission
    // -------------------------------------------------------------
    
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const messageText = chatInput.value.trim();
        if (!messageText) return;

        // Hide welcome state
        if (welcomeContainer) {
            welcomeContainer.classList.add("hidden");
        }

        // Add user message to chat logs
        appendMessage("user", messageText);
        chatInput.value = "";
        chatInput.style.height = "auto";

        // Show typing indicator
        typingIndicator.classList.remove("hidden");
        scrollToBottom();

        try {
            const provider = providerSelect.value;
            const model = modelSelect.value;
            const persona = personaSelect.value;

            const headers = {
                "Content-Type": "application/json"
            };

            const openRouterKey = localStorage.getItem("user_openrouter_key");
            const groqKey = localStorage.getItem("user_groq_key");
            const openAiKey = localStorage.getItem("user_openai_key");
            const geminiKey = localStorage.getItem("user_gemini_key");
            const deepseekKey = localStorage.getItem("user_deepseek_key");

            if (openRouterKey) headers["X-Openrouter-Key"] = openRouterKey;
            if (groqKey) headers["X-Groq-Key"] = groqKey;
            if (openAiKey) headers["X-Openai-Key"] = openAiKey;
            if (geminiKey) headers["X-Gemini-Key"] = geminiKey;
            if (deepseekKey) headers["X-Deepseek-Key"] = deepseekKey;

            const response = await fetch(`${API_BASE}/api/chat`, {
                method: "POST",
                headers: headers,
                body: JSON.stringify({
                    query: messageText,
                    provider: provider,
                    model: model,
                    persona_mode: persona,
                    project_id: currentProjectId
                })
            });

            typingIndicator.classList.add("hidden");

            if (!response.ok) {
                throw new Error(await getErrorMessage(response, "Failed to connect to chatbot"));
            }

            // Create assistant bubble
            const messageDiv = appendMessageDOM("assistant", "");
            const contentDiv = messageDiv.querySelector(".message-content");

            // Read the stream response
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let assistantText = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    // Save completed message to chat history
                    activeChatHistory.push({ sender: "assistant", text: assistantText });
                    localStorage.setItem(`chat_history_${currentProjectId}`, JSON.stringify(activeChatHistory));
                    break;
                }

                const chunkText = decoder.decode(value, { stream: true });
                assistantText += chunkText;

                // Dynamically detect and style streamed errors
                const isError = assistantText.includes("[Error");
                if (isError) {
                    messageDiv.classList.add("error-message");
                    const avatarDiv = messageDiv.querySelector(".message-avatar");
                    if (avatarDiv) avatarDiv.textContent = "⚠️";
                }

                // Render styled markdown and citations
                contentDiv.innerHTML = formatMarkdownCitations(assistantText);
                scrollToBottom();
            }

        } catch (err) {
            typingIndicator.classList.add("hidden");
            appendMessage("assistant", `[Error: ${err.message}]`);
            scrollToBottom();
        }
    });

    // -------------------------------------------------------------
    // Project Switcher & Management Logic
    // -------------------------------------------------------------
    
    async function loadProjects() {
        try {
            const response = await fetch(`${API_BASE}/api/projects`);
            if (!response.ok) throw new Error("Failed to load groups list");
            const projects = await response.json();
            
            groupList.innerHTML = "";
            projects.forEach(proj => {
                const li = document.createElement("li");
                li.className = `group-item${proj === currentProjectId ? ' active' : ''}`;
                li.setAttribute("data-id", proj);
                
                li.innerHTML = `
                    <div class="group-item-clickable">
                        <span class="group-icon">📁</span>
                        <span class="group-name" title="${escapeHtml(proj)}">${escapeHtml(proj)}</span>
                    </div>
                    ${proj !== 'default' ? `
                        <button type="button" class="btn-delete-group" data-id="${escapeHtml(proj)}" title="Delete Group">🗑️</button>
                    ` : ''}
                `;
                
                // Clicking the group name selects it
                li.querySelector(".group-item-clickable").addEventListener("click", () => {
                    selectGroup(proj);
                });
                
                // Wire up delete button if present
                const btnDel = li.querySelector(".btn-delete-group");
                if (btnDel) {
                    btnDel.addEventListener("click", (e) => {
                        e.stopPropagation();
                        deleteGroup(proj);
                    });
                }
                
                groupList.appendChild(li);
            });
            
            if (!projects.includes(currentProjectId)) {
                currentProjectId = projects[0] || "default";
            }
            
            // Sync group header title
            activeProjectName.textContent = currentProjectId;
            
            loadChatHistory();
            loadVideos();
        } catch (err) {
            console.error("Error loading groups: ", err);
        }
    }

    function selectGroup(groupId) {
        if (currentProjectId === groupId) return;
        currentProjectId = groupId;
        activeProjectName.textContent = currentProjectId;
        
        // Update active class in list UI
        const items = groupList.querySelectorAll(".group-item");
        items.forEach(item => {
            if (item.getAttribute("data-id") === groupId) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });
        
        loadChatHistory();
        loadVideos();
    }

    // Inline Group Creator Toggle
    btnAddGroupToggle.addEventListener("click", () => {
        inlineGroupForm.classList.toggle("hidden");
        if (!inlineGroupForm.classList.contains("hidden")) {
            inlineGroupName.value = "";
            inlineGroupName.focus();
        }
    });

    btnInlineGroupCancel.addEventListener("click", () => {
        inlineGroupForm.classList.add("hidden");
        inlineGroupName.value = "";
    });

    // Keydown check for inline creator input
    inlineGroupName.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            saveGroup();
        } else if (e.key === "Escape") {
            inlineGroupForm.classList.add("hidden");
            inlineGroupName.value = "";
        }
    });

    btnInlineGroupSave.addEventListener("click", () => {
        saveGroup();
    });

    async function saveGroup() {
        const name = inlineGroupName.value.trim();
        if (!name) return;
        
        try {
            const response = await fetch(`${API_BASE}/api/projects`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: name })
            });
            
            if (!response.ok) {
                throw new Error(await getErrorMessage(response, "Failed to create group"));
            }
            
            const data = await response.json();
            currentProjectId = data.project_id;
            
            // Reset & hide creator
            inlineGroupForm.classList.add("hidden");
            inlineGroupName.value = "";
            
            loadChatHistory();
            await loadProjects();
        } catch (err) {
            showToast("Group creation error: " + err.message, "error");
        }
    }

    // Delete group handler
    async function deleteGroup(projId) {
        const confirmMsg = projId === "default" 
            ? "Are you sure you want to clear the default group? All its knowledge files will be wiped."
            : `Are you sure you want to delete the group '${projId}' and all its data? This cannot be undone.`;
            
        if (!confirm(confirmMsg)) return;
        
        try {
            const response = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(projId)}`, {
                method: "DELETE"
            });
            
            if (!response.ok) throw new Error("Failed to delete group");
            
            if (currentProjectId === projId) {
                currentProjectId = "default";
            }
            loadChatHistory();
            await loadProjects();
        } catch (err) {
            showToast("Group deletion error: " + err.message, "error");
        }
    }

    // Reset Chat messages log (keeps only welcome screen)
    function resetChatLog() {
        const messages = chatMessages.querySelectorAll(".message");
        messages.forEach(m => m.remove());
        if (welcomeContainer) {
            welcomeContainer.classList.remove("hidden");
        }
    }

    // Load Chat history from localStorage
    function loadChatHistory() {
        resetChatLog();
        const saved = localStorage.getItem(`chat_history_${currentProjectId}`);
        if (saved) {
            try {
                activeChatHistory = JSON.parse(saved);
            } catch (e) {
                activeChatHistory = [];
            }
        } else {
            activeChatHistory = [];
        }
        
        if (activeChatHistory.length > 0) {
            if (welcomeContainer) {
                welcomeContainer.classList.add("hidden");
            }
            activeChatHistory.forEach(msg => {
                appendMessageDOM(msg.sender, msg.text);
            });
        } else {
            if (welcomeContainer) {
                welcomeContainer.classList.remove("hidden");
            }
        }
    }

    // Clear Chat History
    const btnClearChat = document.getElementById("btn-clear-chat");
    if (btnClearChat) {
        btnClearChat.addEventListener("click", () => {
            if (confirm("Are you sure you want to clear the chat history for this project?")) {
                activeChatHistory = [];
                localStorage.removeItem(`chat_history_${currentProjectId}`);
                loadChatHistory();
            }
        });
    }

    // Batch Download transcripts
    const btnBatchDownload = document.getElementById("btn-batch-download");
    if (btnBatchDownload) {
        btnBatchDownload.addEventListener("click", async () => {
            const originalText = btnBatchDownload.innerHTML;
            btnBatchDownload.disabled = true;
            btnBatchDownload.innerHTML = "⏳";
            
            try {
                const response = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(currentProjectId)}/export-batch`);
                if (!response.ok) {
                    throw new Error(await getErrorMessage(response, "Batch export failed"));
                }
                
                const blob = await response.blob();
                const downloadUrl = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = downloadUrl;
                a.download = `transcripts_${currentProjectId}.zip`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(downloadUrl);
            } catch (err) {
                showToast("Batch export error: " + err.message, "error");
            } finally {
                btnBatchDownload.disabled = false;
                btnBatchDownload.innerHTML = originalText;
            }
        });
    }

    // Delete All VIDs in project knowledge base
    const btnDeleteAllVideos = document.getElementById("btn-delete-all-videos");
    if (btnDeleteAllVideos) {
        btnDeleteAllVideos.addEventListener("click", async () => {
            if (confirm("⚠️ WARNING: Are you sure you want to delete ALL videos from the Knowledge Base? This will completely wipe your search index!")) {
                const originalText = btnDeleteAllVideos.innerHTML;
                btnDeleteAllVideos.disabled = true;
                btnDeleteAllVideos.innerHTML = "⏳";
                try {
                    const response = await fetch(`${API_BASE}/api/videos?project_id=${encodeURIComponent(currentProjectId)}`, {
                        method: "DELETE"
                    });
                    if (!response.ok) {
                        throw new Error(await getErrorMessage(response, "Failed to delete all videos"));
                    }
                    showToast("All videos deleted successfully.", "success");
                    loadVideos();
                } catch (err) {
                    showToast("Delete error: " + err.message, "error");
                } finally {
                    btnDeleteAllVideos.disabled = false;
                    btnDeleteAllVideos.innerHTML = originalText;
                }
            }
        });
    }

    // -------------------------------------------------------------
    // Library Video Ingested List Loading & Filter
    // -------------------------------------------------------------
    
    async function loadVideos() {
        try {
            const response = await fetch(`${API_BASE}/api/videos?project_id=${encodeURIComponent(currentProjectId)}`);
            if (!response.ok) throw new Error("Failed to load videos list");
            
            const videos = await response.json();
            
            // Sync count display
            videoCount.textContent = `${videos.length} VID${videos.length === 1 ? '' : 's'}`;
            
            // Clear existing list items (except welcome/empty messages)
            const oldItems = videoList.querySelectorAll(".video-item");
            oldItems.forEach(item => item.remove());

            if (videos.length === 0) {
                emptyVideosMsg.classList.remove("hidden");
                emptyVideosMsg.textContent = "No VIDs ingested yet. Click 'Ingest VIDs' above!";
                return;
            }
            
            emptyVideosMsg.classList.add("hidden");
            
            // Render video library items
            videos.forEach(video => {
                const li = document.createElement("li");
                li.className = "video-item";
                
                li.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 0.95rem;">📹</span>
                        <h4 title="${escapeHtml(video.title)}">${escapeHtml(video.title)}</h4>
                    </div>
                    <div class="video-item-actions">
                        <a href="${video.url}" class="video-item-link" target="_blank" title="Watch Original VID">
                            Original VID 🔗
                        </a>
                        <div class="export-buttons-group">
                            <button class="btn-export-md" data-id="${video.id}" title="Export Raw Transcript to Markdown">
                                Raw MD 📝
                            </button>
                            <button class="btn-export-ai" data-id="${video.id}" title="Generate AI-polished Summary & Transcript">
                                AI Summary ✨
                            </button>
                            <button class="btn-delete-video btn-export-md" data-id="${video.id}" title="Delete Video from Knowledge Base" style="color: var(--text-error); background: transparent; border-color: rgba(239, 68, 68, 0.2); font-weight: bold; padding: 2px 6px;">
                                🗑️
                            </button>
                        </div>
                    </div>
                `;
                videoList.appendChild(li);
            });

            // Re-apply filter if text is in search bar
            if (videoSearch.value) {
                videoSearch.dispatchEvent(new Event("input"));
            }

        } catch (err) {
            console.error("Error loading knowledge base: ", err);
        }
    }

    // Video search filter event listener
    videoSearch.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase().trim();
        const items = videoList.querySelectorAll(".video-item");
        
        items.forEach(item => {
            const title = item.querySelector("h4").textContent.toLowerCase();
            if (title.includes(query)) {
                item.classList.remove("hidden");
            } else {
                item.classList.add("hidden");
            }
        });
        
        // Empty message handler on search mismatch
        const visibleItems = videoList.querySelectorAll(".video-item:not(.hidden)");
        if (visibleItems.length === 0 && items.length > 0) {
            emptyVideosMsg.textContent = "No matching VIDs found.";
            emptyVideosMsg.classList.remove("hidden");
        } else if (items.length > 0) {
            emptyVideosMsg.classList.add("hidden");
        } else {
            emptyVideosMsg.textContent = "No VIDs ingested yet. Click 'Ingest VIDs' above!";
            emptyVideosMsg.classList.remove("hidden");
        }
    });

    // Exporter & Deletion delegations
    videoList.addEventListener("click", async (e) => {
        const btnDelete = e.target.closest(".btn-delete-video");
        if (btnDelete) {
            const videoId = btnDelete.getAttribute("data-id");
            if (confirm("Are you sure you want to delete this video from the Knowledge Base? This will also remove it from the vector search index.")) {
                const originalText = btnDelete.innerHTML;
                btnDelete.disabled = true;
                btnDelete.innerHTML = "⏳";
                try {
                    const response = await fetch(`${API_BASE}/api/videos/${encodeURIComponent(videoId)}?project_id=${encodeURIComponent(currentProjectId)}`, {
                        method: "DELETE"
                    });
                    if (!response.ok) {
                        throw new Error(await getErrorMessage(response, "Failed to delete video"));
                    }
                    showToast("Video deleted successfully.", "success");
                    loadVideos();
                } catch (err) {
                    showToast("Delete error: " + err.message, "error");
                    btnDelete.disabled = false;
                    btnDelete.innerHTML = originalText;
                }
            }
            return;
        }

        const btnRaw = e.target.closest(".btn-export-md");
        const btnAI = e.target.closest(".btn-export-ai");
        
        if (!btnRaw && !btnAI) return;
        
        const btn = btnRaw || btnAI;
        const useAI = !!btnAI;
        const videoId = btn.getAttribute("data-id");
        const originalText = btn.innerHTML;
        
        const videoItem = btn.closest(".video-item");
        const videoTitle = videoItem ? videoItem.querySelector("h4").textContent : "Video";
        const actionsGroup = btn.closest(".video-item-actions");
        
        btn.disabled = true;
        btn.innerHTML = useAI ? "Polishing... ✨" : "Exporting... ⏳";
        if (actionsGroup) actionsGroup.classList.add("loading");
        
        try {
            // Include active provider and model in query parameters for AI polishing
            const provider = providerSelect.value;
            const model = modelSelect.value;
            
            // Get keys from local storage
            const openAiKey = localStorage.getItem("user_openai_key");
            const geminiKey = localStorage.getItem("user_gemini_key");
            const groqKey = localStorage.getItem("user_groq_key");
            const deepseekKey = localStorage.getItem("user_deepseek_key");
            const openRouterKey = localStorage.getItem("user_openrouter_key");

            const url = `${API_BASE}/api/videos/${encodeURIComponent(videoId)}/export?project_id=${encodeURIComponent(currentProjectId)}&use_ai=${useAI}&provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(model)}`;
            
            // Add custom API headers if we have them
            const headers = {};
            if (openAiKey) headers["X-Openai-Key"] = openAiKey;
            if (geminiKey) headers["X-Gemini-Key"] = geminiKey;
            if (groqKey) headers["X-Groq-Key"] = groqKey;
            if (deepseekKey) headers["X-Deepseek-Key"] = deepseekKey;
            if (openRouterKey) headers["X-Openrouter-Key"] = openRouterKey;

            const makeExportRequest = async () => {
                const response = await fetch(url, { headers });
                if (!response.ok) {
                    throw new Error(await getErrorMessage(response, "Export failed"));
                }
                
                // If the backend returned 202, it means the polishing task is running in the background
                if (response.status === 202) {
                    const statusData = await response.json();
                    
                    let activeToast = showToast(`Starting AI Polishing for "${escapeHtml(videoTitle)}"...`, "polish", 30000);
                    
                    // Poll the status API
                    const pollInterval = setInterval(async () => {
                        try {
                            const statusResponse = await fetch(`${API_BASE}/api/videos/${encodeURIComponent(videoId)}/polish-status?project_id=${encodeURIComponent(currentProjectId)}`);
                            if (!statusResponse.ok) return;
                            const status = await statusResponse.json();
                            if (status.status === "completed") {
                                clearInterval(pollInterval);
                                if (activeToast) {
                                    activeToast.update(`AI Polishing complete! Downloading file...`);
                                    setTimeout(() => activeToast.dismiss(), 2000);
                                }
                                // Polishing complete, make export request again to download the cached file
                                makeExportRequest();
                            } else if (status.status === "failed") {
                                clearInterval(pollInterval);
                                if (activeToast) activeToast.dismiss();
                                showToast("AI Polishing failed: " + status.error, "error");
                                btn.disabled = false;
                                btn.innerHTML = originalText;
                                if (actionsGroup) actionsGroup.classList.remove("loading");
                            } else {
                                const percent = Math.round(status.percent);
                                btn.innerHTML = `Polishing... ${percent}% ✨`;
                                if (activeToast) {
                                    activeToast.update(`Polishing "${escapeHtml(videoTitle)}"... <strong>${percent}%</strong>`);
                                }
                            }
                        } catch (pollErr) {
                            console.error("Polling error:", pollErr);
                        }
                    }, 3000);
                    return;
                }
                
                const blob = await response.blob();
                
                // Extract filename from response header
                const disposition = response.headers.get("content-disposition");
                let filename = useAI ? "polished_summary.md" : "transcript.md";
                if (disposition && disposition.indexOf("attachment") !== -1) {
                    const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                    const matches = filenameRegex.exec(disposition);
                    if (matches != null && matches[1]) {
                        filename = matches[1].replace(/['"]/g, '');
                    }
                }
                
                const downloadUrl = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = downloadUrl;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(downloadUrl);
                
                btn.disabled = false;
                btn.innerHTML = originalText;
                if (actionsGroup) actionsGroup.classList.remove("loading");
                showToast(`Downloaded: ${filename}`, "success");
            };
            
            await makeExportRequest();
            
        } catch (err) {
            showToast("Export error: " + err.message, "error");
            btn.disabled = false;
            btn.innerHTML = originalText;
            if (actionsGroup) actionsGroup.classList.remove("loading");
        }
    });

    // -------------------------------------------------------------
    // Chat Message Rendering Helpers
    // -------------------------------------------------------------
    
    // DOM-only append helper
    function appendMessageDOM(sender, text) {
        if (welcomeContainer) {
            welcomeContainer.classList.add("hidden");
        }

        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${sender}-message`;
        
        // Detect error
        const isError = sender === "assistant" && text.includes("[Error");
        if (isError) {
            messageDiv.classList.add("error-message");
        }
        
        const avatarDiv = document.createElement("div");
        avatarDiv.className = "message-avatar";
        avatarDiv.textContent = isError ? "⚠️" : (sender === "user" ? "👤" : "🔮");

        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        contentDiv.innerHTML = formatMarkdownCitations(text);

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        scrollToBottom();

        return messageDiv;
    }

    // Appends message and saves to activeChatHistory and localStorage
    function appendMessage(sender, text) {
        appendMessageDOM(sender, text);
        activeChatHistory.push({ sender, text });
        localStorage.setItem(`chat_history_${currentProjectId}`, JSON.stringify(activeChatHistory));
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Markdown conversion & citations rendering
    function formatMarkdownCitations(text) {
        if (!text) return "";
        
        // Detect error blocks in the text
        const errorIndex = text.indexOf("[Error");
        if (errorIndex !== -1) {
            const normalText = text.substring(0, errorIndex).trim();
            const errorText = text.substring(errorIndex);
            
            let html = "";
            if (normalText) {
                html += formatMarkdownNormal(normalText) + "<br><br>";
            }
            html += formatCleanError(errorText);
            return html;
        }
        
        return formatMarkdownNormal(text);
    }

    function formatMarkdownNormal(text) {
        // Escape HTML
        let clean = escapeHtml(text);
        
        // Convert bold/italic tags
        clean = clean.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        clean = clean.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Match Markdown Links: [Citation Text](URL)
        clean = clean.replace(/\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g, (match, text, url) => {
            return `<a href="${url}" target="_blank" title="Jump to timestamp in VID">⏱️ ${text}</a>`;
        });

        // Convert double newlines to breaks
        clean = clean.replace(/\n\n/g, '<br><br>');
        clean = clean.replace(/\n/g, '<br>');

        return clean;
    }

    function formatCleanError(text) {
        // Strip outer brackets if present: "[Error: msg]" or "[Error generating response: msg]"
        let errorMsg = text.trim();
        if (errorMsg.startsWith("[")) {
            // Remove leading '[' and trailing ']' if present
            const match = errorMsg.match(/^\[Error(?: generating response)?:\s*([\s\S]*?)\]?$/i);
            if (match) {
                errorMsg = match[1].trim();
            } else if (errorMsg.endsWith("]")) {
                errorMsg = errorMsg.substring(1, errorMsg.length - 1).trim();
            }
        }
        
        let cleanMsg = escapeHtml(errorMsg);
        let title = "Response Generation Failed";
        let description = "";
        let details = "";
        
        const lowerMsg = cleanMsg.toLowerCase();
        
        // Check for Gemini/Google API Quota Error
        if (lowerMsg.includes("429") || lowerMsg.includes("quota")) {
            title = "API Quota Exceeded (429)";
            description = "You have exceeded the request rate limits or token quota for this model. Please wait a minute before retrying, or switch to a different model/provider.";
            details = cleanMsg;
        } 
        // Check for Model Decommissioned / Model Not Found / 404
        else if (lowerMsg.includes("decommissioned") || lowerMsg.includes("not found") || lowerMsg.includes("404")) {
            title = "AI Model Deprecated or Not Found";
            description = "The selected model is currently decommissioned, deprecated, or not found. Please select a different active model from the dropdown list in the top header.";
            details = cleanMsg;
        }
        // General API key / credentials errors
        else if (lowerMsg.includes("api_key") || lowerMsg.includes("api key") || lowerMsg.includes("unauthorized") || lowerMsg.includes("401")) {
            title = "API Authentication Failed (401)";
            description = "Authentication with the AI provider failed. Please verify that your API key is correctly configured in your server `.env` settings.";
            details = cleanMsg;
        }
        else {
            // General error
            title = "AI Engine Error";
            if (cleanMsg.length < 180) {
                description = cleanMsg;
            } else {
                description = "The chatbot engine encountered an error while processing the request. Expand the technical details below for more information.";
                details = cleanMsg;
            }
        }
        
        // Build styled HTML alert
        let html = `
            <div class="error-bubble-container">
                <div class="error-bubble-header">
                    <span class="error-bubble-title">⚠️ ${title}</span>
                </div>
                <div class="error-bubble-body">
                    <p class="error-bubble-desc">${description}</p>
                </div>
        `;
        
        if (details) {
            // Make any URLs inside the technical details clickable
            let clickableDetails = details.replace(/(https?:\/\/[^\s\)]+)/g, '<a href="$1" target="_blank">$1</a>');
            html += `
                <details class="error-bubble-details">
                    <summary>Show Technical Details</summary>
                    <pre>${clickableDetails}</pre>
                </details>
            `;
        }
        
        html += `</div>`;
        return html;
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    // Helper: Safely parse error messages from server responses
    async function getErrorMessage(response, defaultMsg) {
        try {
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                const errData = await response.json();
                return errData.detail || defaultMsg;
            } else {
                const txt = await response.text();
                return txt || response.statusText || defaultMsg;
            }
        } catch (e) {
            return response.statusText || defaultMsg;
        }
    }

    // Modern Toast Notification UI Helper
    function showToast(message, type = "info", duration = 4000) {
        let container = document.getElementById("toast-container");
        if (!container) {
            container = document.createElement("div");
            container.id = "toast-container";
            document.body.appendChild(container);
        }
        
        const toast = document.createElement("div");
        toast.className = `toast-message ${type}`;
        
        let icon = "ℹ️";
        if (type === "success") icon = "✅";
        if (type === "error") icon = "⚠️";
        if (type === "polish") icon = "✨";
        
        toast.innerHTML = `<span class="toast-icon">${icon}</span><span class="toast-text">${message}</span>`;
        container.appendChild(toast);
        
        // Trigger reflow & show
        setTimeout(() => toast.classList.add("show"), 10);
        
        // Hide and remove helper
        const removeToast = () => {
            toast.classList.remove("show");
            toast.classList.add("hide");
            setTimeout(() => toast.remove(), 300);
        };
        
        if (duration > 0) {
            setTimeout(removeToast, duration);
        }
        
        return {
            element: toast,
            update: (newMessage) => {
                const textSpan = toast.querySelector(".toast-text");
                if (textSpan) textSpan.innerHTML = newMessage;
            },
            dismiss: removeToast
        };
    }
});
