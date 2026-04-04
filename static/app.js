document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.querySelector("#image");
    const fileName = document.querySelector("[data-file-name]");
    const loadingOverlay = document.querySelector("[data-loading]");
    const loadingText = document.querySelector("[data-loading-text]");
    const forms = document.querySelectorAll("[data-load-form]");

    const startCameraButton = document.querySelector("[data-start-camera]");
    const stopCameraButton = document.querySelector("[data-stop-camera]");
    const webcamStage = document.querySelector(".webcam-stage");
    const webcamVideo = document.querySelector("#webcamVideo");
    const webcamOverlay = document.querySelector("#webcamOverlay");
    const cameraPlaceholder = document.querySelector("[data-camera-placeholder]");
    const cameraStatus = document.querySelector("[data-camera-status]");
    const liveFaceCount = document.querySelector("[data-live-face-count]");
    const liveLatency = document.querySelector("[data-live-latency]");
    const liveFaceEmotion = document.querySelector("[data-live-face-emotion]");
    const liveTextResult = document.querySelector("[data-live-text-result]");
    const liveStressLevel = document.querySelector("[data-live-stress-level]");
    const liveEmpty = document.querySelector("[data-live-empty]");
    const liveFaceList = document.querySelector("[data-live-face-list]");
    const standaloneStressText = document.querySelector("#standaloneStressText");
    const analyzeTextButton = document.querySelector("[data-analyze-text]");
    const textResultEmpty = document.querySelector("[data-text-result-empty]");
    const textResultMessage = document.querySelector("[data-text-result-message]");
    const textResultSummary = document.querySelector("[data-text-result-summary]");
    const textResultLabel = document.querySelector("[data-text-result-label]");
    const textResultConfidence = document.querySelector("[data-text-result-confidence]");
    const textResultLevel = document.querySelector("[data-text-result-level]");
    const textSupportCard = document.querySelector("[data-text-support-card]");
    const textSupportMessage = document.querySelector("[data-text-support-message]");
    const textSuggestionsCard = document.querySelector("[data-text-suggestions-card]");
    const textSuggestions = document.querySelector("[data-text-suggestions]");

    const MAX_FRAME_WIDTH = 640;
    const REQUEST_INTERVAL_MS = 140;

    const state = {
        stream: null,
        running: false,
        requestInFlight: false,
        timerId: null,
        sessionId: null,
        processingCanvas: document.createElement("canvas"),
    };

    const processingContext = state.processingCanvas.getContext("2d");
    const overlayContext = webcamOverlay ? webcamOverlay.getContext("2d") : null;

    function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, (character) => {
            const replacements = {
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                "\"": "&quot;",
                "'": "&#39;",
            };
            return replacements[character] || character;
        });
    }

    function createSessionId() {
        return `stream-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    }

    function setButtonState() {
        if (startCameraButton) {
            startCameraButton.disabled = state.running;
        }

        if (stopCameraButton) {
            stopCameraButton.disabled = !state.running;
        }
    }

    function setCameraStatus(text) {
        if (cameraStatus) {
            cameraStatus.textContent = text;
        }
    }

    function updateLiveStats(faceCount, latencyText, faceEmotionText, textResultText, stressLevelText) {
        if (liveFaceCount) {
            liveFaceCount.textContent = String(faceCount);
        }

        if (liveLatency) {
            liveLatency.textContent = latencyText;
        }

        if (liveFaceEmotion) {
            liveFaceEmotion.textContent = faceEmotionText;
        }

        if (liveTextResult) {
            liveTextResult.textContent = textResultText;
        }

        if (liveStressLevel) {
            liveStressLevel.textContent = stressLevelText;
        }
    }

    function setStageActive(isActive) {
        if (webcamStage) {
            webcamStage.classList.toggle("active", isActive);
        }

        if (cameraPlaceholder) {
            cameraPlaceholder.hidden = isActive;
        }
    }

    function showLiveEmpty(message) {
        if (!liveEmpty) {
            return;
        }

        const paragraph = liveEmpty.querySelector("p");
        if (paragraph) {
            paragraph.textContent = message;
        } else {
            liveEmpty.textContent = message;
        }

        liveEmpty.hidden = false;
    }

    function clearLiveFaceList() {
        if (liveFaceList) {
            liveFaceList.innerHTML = "";
            liveFaceList.hidden = true;
        }
    }

    function setTextResultState({ message = "", isError = false, result = null }) {
        if (textResultEmpty) {
            textResultEmpty.hidden = Boolean(message) || Boolean(result);
        }

        if (textResultMessage) {
            if (message) {
                textResultMessage.hidden = false;
                textResultMessage.textContent = message;
                textResultMessage.classList.toggle("error", isError);
            } else {
                textResultMessage.hidden = true;
                textResultMessage.textContent = "";
                textResultMessage.classList.remove("error");
            }
        }

        if (textResultSummary) {
            textResultSummary.hidden = !result;
        }

        if (textSupportCard) {
            textSupportCard.hidden = !result || !result.support_message;
        }

        if (textSupportMessage) {
            textSupportMessage.textContent = result && result.support_message
                ? result.support_message
                : "";
        }

        if (textSuggestionsCard) {
            textSuggestionsCard.hidden = !result || !Array.isArray(result.suggestions) || result.suggestions.length === 0;
        }

        if (textSuggestions) {
            textSuggestions.innerHTML = result && Array.isArray(result.suggestions)
                ? result.suggestions.map((suggestion) => `<li>${escapeHtml(suggestion)}</li>`).join("")
                : "";
        }

        if (result) {
            const confidenceText = result.confidence_text
                || (typeof result.confidence === "number" ? `${(result.confidence * 100).toFixed(1)}%` : "Unknown");
            const statusText = result.status
                || result.display_name
                || (typeof result.prediction === "number"
                    ? (result.prediction === 1 ? "Stressed" : "Not Stressed")
                    : result.label || "Unknown");

            if (textResultLabel) {
                textResultLabel.textContent = statusText;
            }
            if (textResultConfidence) {
                const confValue = (result.confidence || 0) * 100;
                let confLevel = "Strong";
                let confColor = "#22c55e";
                let showHint = false;
                
                if (confValue >= 80) {
                    confLevel = "Strong";
                    confColor = "#22c55e";
                } else if (confValue >= 65) {
                    confLevel = "Moderate";
                    confColor = "#f97316";
                } else {
                    confLevel = "Low &mdash; try a longer sentence";
                    confColor = "#ef4444";
                    showHint = true;
                }

                textResultConfidence.innerHTML = `<span style="color: ${confColor}">${confidenceText}</span>`;
                
                const siblingSmall = textResultConfidence.nextElementSibling;
                if (siblingSmall && siblingSmall.tagName === 'SMALL') {
                    siblingSmall.innerHTML = confLevel;
                }
                
                const hintDiv = document.querySelector('[data-text-result-hint]');
                if (hintDiv) {
                    hintDiv.hidden = !showHint;
                }
            }
            if (textResultLevel) {
                textResultLevel.textContent = result.stress || result.stress_level_display || result.stress_level || "Unknown";
            }
        }
    }

    function renderLiveFaceCards(detections) {
        if (!liveFaceList || !liveEmpty) {
            return;
        }

        if (!detections || detections.length === 0) {
            clearLiveFaceList();
            showLiveEmpty("Camera is running. No face is currently detected, but text analysis can still continue.");
            return;
        }

        liveEmpty.hidden = true;
        liveFaceList.hidden = false;
        liveFaceList.innerHTML = detections.map((detection, index) => `
            <div class="face-card">
                <div class="face-card-head">
                    <strong>Face ${index + 1}</strong>
                    <span>${escapeHtml(detection.result_display)}</span>
                </div>
                <p class="face-confidence">Stable final output for live demo mode</p>
            </div>
        `).join("");
    }

    function resetOverlay() {
        if (overlayContext && webcamOverlay) {
            overlayContext.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);
        }
    }

    function syncCanvasDimensions() {
        if (!webcamVideo || !webcamOverlay || !state.processingCanvas) {
            return null;
        }

        if (!webcamVideo.videoWidth || !webcamVideo.videoHeight) {
            return null;
        }

        const scale = Math.min(1, MAX_FRAME_WIDTH / webcamVideo.videoWidth);
        const width = Math.max(1, Math.round(webcamVideo.videoWidth * scale));
        const height = Math.max(1, Math.round(webcamVideo.videoHeight * scale));

        if (state.processingCanvas.width !== width || state.processingCanvas.height !== height) {
            state.processingCanvas.width = width;
            state.processingCanvas.height = height;
        }

        if (webcamOverlay.width !== width || webcamOverlay.height !== height) {
            webcamOverlay.width = width;
            webcamOverlay.height = height;
        }

        return { width, height };
    }

    function captureFrame() {
        const dimensions = syncCanvasDimensions();
        if (!dimensions || !processingContext) {
            return null;
        }

        processingContext.drawImage(webcamVideo, 0, 0, dimensions.width, dimensions.height);

        return {
            width: dimensions.width,
            height: dimensions.height,
            image: state.processingCanvas.toDataURL("image/jpeg", 0.72),
        };
    }

    function drawDetectionLabel(label, x, y, color) {
        if (!overlayContext) {
            return;
        }

        overlayContext.font = "600 16px Manrope, sans-serif";
        overlayContext.textBaseline = "top";

        const textWidth = overlayContext.measureText(label).width;
        const paddingX = 10;
        const paddingY = 7;
        const labelWidth = textWidth + paddingX * 2;
        const labelHeight = 18 + paddingY * 2;
        const labelX = x;
        const labelY = Math.max(8, y - labelHeight - 8);

        overlayContext.fillStyle = color;
        overlayContext.fillRect(labelX, labelY, labelWidth, labelHeight);

        overlayContext.fillStyle = "#08121f";
        overlayContext.fillText(label, labelX + paddingX, labelY + paddingY);
    }

    function drawOverlay(detections) {
        if (!overlayContext || !webcamOverlay) {
            return;
        }

        resetOverlay();

        if (!detections || detections.length === 0) {
            return;
        }

        overlayContext.lineWidth = 3;
        overlayContext.lineJoin = "round";

        detections.forEach((detection) => {
            const [x, y, width, height] = detection.bbox;
            const color = detection.is_sure ? "#2dd4bf" : "#f9c74f";
            const label = detection.result_display;

            overlayContext.strokeStyle = color;
            overlayContext.strokeRect(x, y, width, height);
            drawDetectionLabel(label, x, y, color);
        });
    }

    function applyLivePrediction(payload, fallbackLatencyMs) {
        const detections = payload.detections || [];
        const primary = payload.primary;
        const textResult = payload.text_result;
        const multimodalResult = payload.multimodal_result;
        const latencyValue = Number(payload.server_latency_ms || fallbackLatencyMs || 0);
        const latencyText = `${latencyValue.toFixed(0)} ms`;
        const faceEmotionText = primary
            ? primary.result_display
            : "No face";
        const textResultText = textResult
            ? (textResult.status || textResult.display_name || textResult.result_display)
            : "No text";
        const stressLevelText = multimodalResult
            ? multimodalResult.result_display
            : "Waiting";

        if (payload.success) {
            setCameraStatus("Detecting");
        } else {
            setCameraStatus("No Face");
        }

        updateLiveStats(
            detections.length,
            latencyText,
            faceEmotionText,
            textResultText,
            stressLevelText,
        );
        drawOverlay(detections);
        renderLiveFaceCards(detections);
    }

    async function notifyStopStream() {
        if (!state.sessionId) {
            return;
        }

        try {
            await fetch("/stop-stream", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ stream_id: state.sessionId }),
                keepalive: true,
            });
        } catch (error) {
            console.warn("Could not notify Flask to stop the stream session.", error);
        }
    }

    async function processFrameLoop() {
        if (!state.running) {
            return;
        }

        if (document.hidden) {
            state.timerId = window.setTimeout(processFrameLoop, REQUEST_INTERVAL_MS * 2);
            return;
        }

        if (state.requestInFlight) {
            state.timerId = window.setTimeout(processFrameLoop, REQUEST_INTERVAL_MS);
            return;
        }

        const capturedFrame = captureFrame();
        if (!capturedFrame) {
            state.timerId = window.setTimeout(processFrameLoop, REQUEST_INTERVAL_MS);
            return;
        }

        state.requestInFlight = true;
        const requestStartedAt = performance.now();

        try {
            const response = await fetch("/predict-frame", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    image: capturedFrame.image,
                    stream_id: state.sessionId,
                }),
            });

            const payload = await response.json();

            if (!state.running) {
                return;
            }

            if (!response.ok) {
                throw new Error(payload.message || "Live prediction request failed.");
            }

            applyLivePrediction(payload, performance.now() - requestStartedAt);
        } catch (error) {
            if (state.running) {
                setCameraStatus("Connection Error");
                updateLiveStats(0, "--", "Unavailable", "Unavailable", "Unavailable");
                resetOverlay();
                clearLiveFaceList();
                showLiveEmpty(error.message || "Could not process live webcam frames.");
            }
        } finally {
            state.requestInFlight = false;

            if (state.running) {
                state.timerId = window.setTimeout(processFrameLoop, REQUEST_INTERVAL_MS);
            }
        }
    }

    async function analyzeTextOnly() {
        if (!standaloneStressText || !analyzeTextButton) {
            return;
        }

        const userText = standaloneStressText.value.trim();
        if (!userText) {
            setTextResultState({
                message: "Please enter some text before running text analysis.",
                isError: true,
                result: null,
            });
            standaloneStressText.focus();
            return;
        }

        analyzeTextButton.disabled = true;
        const originalButtonText = analyzeTextButton.textContent;
        analyzeTextButton.textContent = "Analyzing...";

        try {
            const response = await fetch("/predict-text", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text: userText }),
            });

            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.message || payload.error || "Text analysis failed.");
            }

            const textResult = payload.text_result || payload;

            setTextResultState({
                message: "Text analysis completed successfully.",
                isError: false,
                result: textResult,
            });
        } catch (error) {
            setTextResultState({
                message: error.message || "Could not analyze the text right now.",
                isError: true,
                result: null,
            });
        } finally {
            analyzeTextButton.disabled = false;
            analyzeTextButton.textContent = originalButtonText;
        }
    }

    async function startCamera() {
        if (state.running) {
            return;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setCameraStatus("Unsupported");
            showLiveEmpty("This browser does not support webcam access with getUserMedia.");
            return;
        }

        try {
            setCameraStatus("Starting");
            showLiveEmpty("Requesting webcam permission from the browser...");

            state.sessionId = createSessionId();
            state.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user",
                },
                audio: false,
            });

            webcamVideo.srcObject = state.stream;
            await webcamVideo.play();

            state.running = true;
            setButtonState();
            setStageActive(true);
            setCameraStatus("Camera Live");
            updateLiveStats(
                0,
                "0 ms",
                "Searching",
                "Not used",
                "Waiting",
            );
            showLiveEmpty("Camera started. Looking for faces in the current frame...");

            window.clearTimeout(state.timerId);
            state.timerId = window.setTimeout(processFrameLoop, 180);
        } catch (error) {
            state.running = false;
            setButtonState();
            setStageActive(false);
            setCameraStatus("Permission Needed");

            const readableMessage = error && error.name === "NotAllowedError"
                ? "Camera permission was denied. Allow camera access in the browser and try again."
                : "Could not start the webcam. Check camera permissions and availability.";

            showLiveEmpty(readableMessage);
            updateLiveStats(0, "--", "Unavailable", "Unavailable", "Unavailable");
        }
    }

    async function stopCamera() {
        state.running = false;
        state.requestInFlight = false;

        if (state.timerId) {
            window.clearTimeout(state.timerId);
            state.timerId = null;
        }

        if (state.stream) {
            state.stream.getTracks().forEach((track) => track.stop());
            state.stream = null;
        }

        if (webcamVideo) {
            webcamVideo.pause();
            webcamVideo.srcObject = null;
        }

        setStageActive(false);
        setButtonState();
        setCameraStatus("Idle");
        updateLiveStats(0, "0 ms", "Waiting", "Waiting", "Waiting");
        resetOverlay();
        clearLiveFaceList();
        showLiveEmpty("Live detections will appear here after the browser camera starts.");
        await notifyStopStream();
        state.sessionId = null;
    }

    if (fileInput && fileName) {
        fileInput.addEventListener("change", () => {
            const selectedFile = fileInput.files && fileInput.files.length > 0
                ? fileInput.files[0].name
                : "JPG, PNG, BMP, or WEBP up to 8 MB";
            fileName.textContent = selectedFile;
        });
    }

    forms.forEach((form) => {
        form.addEventListener("submit", () => {
            const activeButton = document.activeElement;
            const message = activeButton && activeButton.dataset.loadingMessage
                ? activeButton.dataset.loadingMessage
                : "Processing request...";

            if (loadingText) {
                loadingText.textContent = message;
            }

            if (loadingOverlay) {
                loadingOverlay.classList.add("visible");
            }
        });
    });

    if (startCameraButton) {
        startCameraButton.addEventListener("click", startCamera);
    }

    if (stopCameraButton) {
        stopCameraButton.addEventListener("click", stopCamera);
    }

    if (analyzeTextButton) {
        analyzeTextButton.addEventListener("click", analyzeTextOnly);
    }

    if (standaloneStressText) {
        standaloneStressText.addEventListener("keydown", (event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();
                analyzeTextOnly();
            }
        });
    }

    window.addEventListener("beforeunload", () => {
        if (state.stream) {
            state.stream.getTracks().forEach((track) => track.stop());
        }
    });

    setButtonState();

    // Physiological Stress Detection
    const hrSlider = document.getElementById('hr-slider');
    const edaSlider = document.getElementById('eda-slider');
    const respSlider = document.getElementById('resp-slider');
    const tempSlider = document.getElementById('temp-slider');

    const hrVal = document.getElementById('hr-val');
    const edaVal = document.getElementById('eda-val');
    const respVal = document.getElementById('resp-val');
    const tempVal = document.getElementById('temp-val');
    
    const btnPredictBio = document.getElementById('predict-bio-btn');
    const bioResultBox = document.getElementById('bio-result-box');
    const bioStressLabel = document.getElementById('bio-stress-label');
    const bioConfidenceLabel = document.getElementById('bio-confidence-label');

    if (hrSlider) {
        const updateVals = () => {
            hrVal.textContent = hrSlider.value;
            edaVal.textContent = edaSlider.value;
            respVal.textContent = respSlider.value;
            tempVal.textContent = tempSlider.value;
        };

        hrSlider.addEventListener('input', updateVals);
        edaSlider.addEventListener('input', updateVals);
        respSlider.addEventListener('input', updateVals);
        tempSlider.addEventListener('input', updateVals);

        btnPredictBio.addEventListener('click', async () => {
            const origText = btnPredictBio.textContent;
            btnPredictBio.textContent = 'Predicting...';
            btnPredictBio.disabled = true;

            try {
                const response = await fetch('/predict-bio', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        heart_rate: parseFloat(hrSlider.value),
                        eda: parseFloat(edaSlider.value),
                        respiration: parseFloat(respSlider.value),
                        temperature: parseFloat(tempSlider.value)
                    })
                });

                const payload = await response.json();

                if (!response.ok) {
                    throw new Error(payload.error || 'Failed to predict bio stress');
                }

                // Log API response in console
                console.log("API response:", payload);

                const statusVal = payload.status || "N/A";
                const confidenceVal = payload.confidence !== undefined ? payload.confidence : "N/A";
                const messageVal = payload.message || "N/A";

                bioConfidenceLabel.innerHTML = `Confidence: ${confidenceVal}%<br><span style="display:inline-block; margin-top:0.5rem; font-size:0.95rem; color:var(--text-secondary, #9ca3af);">Suggestion: ${messageVal}</span>`;
                
                if (statusVal === 'Stressed') {
                    bioStressLabel.textContent = statusVal + ' 🔴';
                    bioStressLabel.style.color = '#ef4444'; // Red
                    bioResultBox.style.borderColor = '#ef4444';
                    bioResultBox.style.backgroundColor = 'rgba(239, 68, 68, 0.05)';
                } else if (statusVal === 'Not Stressed') {
                    bioStressLabel.textContent = statusVal + ' 🟢';
                    bioStressLabel.style.color = '#22c55e'; // Green
                    bioResultBox.style.borderColor = '#22c55e';
                    bioResultBox.style.backgroundColor = 'rgba(34, 197, 94, 0.05)';
                } else {
                    bioStressLabel.textContent = statusVal;
                    bioStressLabel.style.color = 'inherit';
                    bioResultBox.style.borderColor = 'currentColor';
                    bioResultBox.style.backgroundColor = 'transparent';
                }

            } catch (error) {
                console.error(error);
                bioStressLabel.textContent = 'Error';
                bioStressLabel.style.color = 'inherit';
                bioConfidenceLabel.textContent = error.message;
                bioResultBox.style.borderColor = 'currentColor';
                bioResultBox.style.backgroundColor = 'transparent';
            } finally {
                btnPredictBio.textContent = origText;
                btnPredictBio.disabled = false;
            }
        });
    }
});
