// Import the necessary components from Transformers.js
import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0";

// --- DOM Elements ---
// (Keep DOM element selectors as they were)
const modelNameInput = document.getElementById('model-name');
const loadModelButton = document.getElementById('load-model-btn');
const statusDisplay = document.getElementById('status');
const chatHistory = document.getElementById('chat-history');
const userMessageInput = document.getElementById('user-message');
const sendMessageButton = document.getElementById('send-btn');
const systemMessageDiv = document.querySelector('.message.system');

// --- State Variables ---
// (Keep state variables as they were)
let generator = null;
let currentModelName = null;
let isLoading = false;
let isGenerating = false;
let conversationHistory = [];

// --- Helper Functions ---
// (Keep updateStatus, addMessageToChat, handleProgress as they were)
function updateStatus(text, isError = false) {
    statusDisplay.textContent = `Status: ${text}`;
    statusDisplay.style.color = isError ? 'red' : '#555';
    console.log(`Status: ${text}`);
}

function addMessageToChat(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', role);

    const roleSpan = document.createElement('span');
    roleSpan.textContent = role.charAt(0).toUpperCase() + role.slice(1) + ':';
    messageDiv.appendChild(roleSpan);

    if (role === 'assistant') {
         const contentSpan = document.createElement('span');
         contentSpan.classList.add('content');
         contentSpan.textContent = content;
         messageDiv.appendChild(contentSpan);
         // console.log(">>> DEBUG: Created assistant message div with .content span:", messageDiv);
    } else {
        messageDiv.appendChild(document.createTextNode(' ' + content));
    }

    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return messageDiv;
}

function handleProgress(data) {
    let file = data.file || 'file';
    switch (data.status) {
        case 'initiate': updateStatus(`Initiating model loading...`); break;
        case 'progress':
            const percentage = data.total ? (data.progress / data.total * 100).toFixed(1) : '??';
            const loadedMB = (data.loaded / 1024 / 1024).toFixed(1);
            const totalMB = data.total ? (data.total / 1024 / 1024).toFixed(1) : '??';
            updateStatus(`Downloading ${file}... ${percentage}% (${loadedMB}MB / ${totalMB}MB)`);
            break;
        case 'ready': updateStatus(`Model file ${file} ready.`); break;
        case 'done': updateStatus(`Model file ${file} downloaded successfully.`); break;
        case 'loaded': updateStatus(`Model weights loaded.`); break;
        default: updateStatus(`Loading status: ${data.status}` + (data.file ? ` (${data.file})` : ''));
    }
}
// --- MODIFIED STREAMER CLASS ---
class UiTextStreamer {
    constructor(tokenizer, uiUpdateCallback, options = {}) {
        this.tokenizer = tokenizer;
        this.uiUpdateCallback = uiUpdateCallback;
        this.options = {
            skip_special_tokens: true,
            ...options
        };
        this.fullOutput = "";
        this.isDecoding = false;
        // --- NEW --- Define the marker indicating the start of the actual response
        // This usually corresponds to the role name used in the prompt format.
        // Adjust this if your model uses a different format (e.g., "<|im_start|>assistant\n")
        this.assistantResponseMarker = "assistant\n";
        this.foundMarker = false; // Flag to track if we've passed the marker

        console.log(">>> DEBUG: UiTextStreamer instance created (Direct Implementation). Options:", this.options);
    }

    async put(token_ids_array) {
        if (this.isDecoding) return;
        this.isDecoding = true;

        const token_ids = token_ids_array.flat();
        if (!token_ids || token_ids.length === 0) {
            this.isDecoding = false;
            return;
        }

        try {
            const decoded_text = this.tokenizer.decode(token_ids, {
                skip_special_tokens: this.options.skip_special_tokens
            });

            if (decoded_text != null && decoded_text !== '') {
                this.fullOutput += decoded_text; // Append everything received

                let textToDisplay = "";
                if (!this.foundMarker) {
                    // Search for the marker in the *entire* accumulated output so far
                    const markerIndex = this.fullOutput.lastIndexOf(this.assistantResponseMarker);
                    if (markerIndex !== -1) {
                        // Marker found! Extract text *after* the marker.
                        textToDisplay = this.fullOutput.substring(markerIndex + this.assistantResponseMarker.length);
                        this.foundMarker = true; // Set the flag
                        console.log(">>> DEBUG: Assistant marker found. Starting UI stream.");
                    } else {
                         // Marker not found yet, don't display anything (or maybe '...' placeholder)
                         // console.log(">>> DEBUG: Waiting for assistant marker..."); // Optional log
                    }
                } else {
                    // Marker was found previously, so the relevant part is *after* the marker
                    // Find the marker again (safer in case of weird outputs) and take text after it
                     const markerIndex = this.fullOutput.lastIndexOf(this.assistantResponseMarker);
                     if (markerIndex !== -1) {
                         textToDisplay = this.fullOutput.substring(markerIndex + this.assistantResponseMarker.length);
                     } else {
                         // Should not happen if foundMarker is true, but as fallback show fullOutput
                         console.warn(">>> DEBUG WARNING: foundMarker=true but marker disappeared?");
                         textToDisplay = this.fullOutput;
                     }
                }

                // Only call UI update if we have found the marker and have text to show
                if (this.foundMarker && this.uiUpdateCallback) {
                    this.uiUpdateCallback(textToDisplay, false);
                }
            }
        } catch (error) {
             console.error(">>> DEBUG ERROR: Error during token decoding/filtering in put():", error);
        } finally {
            this.isDecoding = false;
        }
    }

    end() {
        console.log(">>> DEBUG: Streamer end() called directly. Final Raw Output:", this.fullOutput);

        let finalTextToDisplay = "";
        // Ensure the final text is also filtered correctly
        if (this.foundMarker) {
             const markerIndex = this.fullOutput.lastIndexOf(this.assistantResponseMarker);
             if (markerIndex !== -1) {
                 finalTextToDisplay = this.fullOutput.substring(markerIndex + this.assistantResponseMarker.length);
             } else {
                 finalTextToDisplay = this.fullOutput; // Fallback
             }
        } else {
            // If marker was *never* found, maybe the model didn't follow the format?
            // Log a warning and maybe display the raw output for debugging.
            console.warn(">>> DEBUG WARNING: Assistant marker was never found in the output stream.");
            // Decide what to display: raw output, or nothing, or an error?
            // Let's show the raw output in this edge case for debugging.
            finalTextToDisplay = this.fullOutput;
        }


        if (this.uiUpdateCallback) {
            this.uiUpdateCallback(finalTextToDisplay, true); // Final update with filtered text
        }

        // Reset flag for the next generation sequence
        this.foundMarker = false;
    }
}

// --- Core Logic Functions ---
// (Keep loadModel as it was in the previous correct version)
async function loadModel() {
    const modelName = modelNameInput.value.trim();
    if (!modelName || isLoading) return;

    if (systemMessageDiv) systemMessageDiv.style.display = 'none';

    isLoading = true;
    generator = null;
    setLoadingState(true);
    updateStatus(`Attempting to load model: ${modelName}...`);
    addMessageToChat('system', `Loading model: ${modelName}...`);
    console.log(">>> DEBUG: Starting model load sequence.");

    try {
        generator = await pipeline("text-generation", modelName, {
            dtype: "q4f16", device: "webgpu", progress_callback: handleProgress,
        });

        if (!generator || !generator.tokenizer) {
            throw new Error("Pipeline call completed but generator or tokenizer is invalid.");
        }
        console.log(">>> DEBUG: Generator and tokenizer loaded successfully.");

        currentModelName = modelName;
        updateStatus(`Model "${modelName}" loaded successfully. Ready to chat.`);
        addMessageToChat('system', `Model "${modelName}" loaded. You can start chatting.`);
        isLoading = false;
        setGeneratingState(false, true); // Enable chat
        console.log(">>> DEBUG: Model load successful. Chat enabled.");

    } catch (error) {
        console.error(">>> DEBUG: Model loading failed:", error);
        updateStatus(`Error loading model: ${error.message || error}`, true);
        addMessageToChat('error', `Failed to load model "${modelName}". Check console. Error: ${error.message || error}`);
        generator = null; currentModelName = null; isLoading = false;
        setLoadingState(false); // Re-enable load controls
        setGeneratingState(false, false); // Keep chat disabled
    }
}

// (Keep generateResponse largely the same, ensuring it uses the new streamer)
async function generateResponse() {
    const userMessage = userMessageInput.value.trim();
    if (!userMessage || isGenerating || isLoading || !generator) return;

    console.log(">>> DEBUG: Starting generateResponse sequence.");
    isGenerating = true;
    setGeneratingState(true);
    addMessageToChat('user', userMessage);
    userMessageInput.value = '';

    conversationHistory.push({ role: "user", content: userMessage });

    const assistantMsgDiv = addMessageToChat('assistant', '...');
    const assistantContentSpan = assistantMsgDiv.querySelector('.content');

    if (!assistantContentSpan) {
        console.error(">>> FATAL DEBUG ERROR: Could not find '.content' span.");
        addMessageToChat('error', "Internal UI error: Cannot create response area.");
        isGenerating = false; setGeneratingState(false); return;
    }
    // console.log(">>> DEBUG: Found '.content' span for UI updates:", assistantContentSpan);

    const updateUICallback = (text, isFinal) => {
        // console.log(`>>> DEBUG: updateUICallback called. Final=${isFinal}. Text=`, text.slice(0,50)+'...');
        if (!assistantContentSpan) return; // Paranoid check
        assistantContentSpan.textContent = text;
        chatHistory.scrollTop = chatHistory.scrollHeight;

        if (isFinal) {
             console.log(">>> DEBUG: Processing final update in UI callback.");
             // Prevent duplicates if end() causes multiple final calls (shouldn't happen now)
             if (!conversationHistory.length || conversationHistory[conversationHistory.length - 1].content !== text || conversationHistory[conversationHistory.length - 1].role !== 'assistant') {
                conversationHistory.push({ role: "assistant", content: text });
             }
             isGenerating = false;
             setGeneratingState(false); // Re-enable inputs
             updateStatus("Ready.");
             console.log(">>> DEBUG: Final Assistant Response logged by UI Callback:", text);
        }
    };

     if (!generator.tokenizer) {
        console.error(">>> FATAL DEBUG ERROR: Tokenizer missing!");
        addMessageToChat('error', "Internal setup error: Tokenizer missing.");
        isGenerating = false; setGeneratingState(false); return;
     }

    // Create the MODIFIED streamer instance
    const streamer = new UiTextStreamer(generator.tokenizer, updateUICallback, {
        skip_prompt: true,          // Pass options here - streamer constructor will store them
        skip_special_tokens: true,
    });
    console.log(">>> DEBUG: UiTextStreamer instance (direct) passed to pipeline:", streamer);

    updateStatus("Generating response...");

    try {
         const messagesForModel = [...conversationHistory];
         if (!messagesForModel.some(m => m.role === 'system')) {
             messagesForModel.unshift({ role: "system", content: "You are a helpful assistant." });
         }
         // console.log(">>> DEBUG: Messages sent to model:", messagesForModel);

        // Generate response using the streamer
         await generator(messagesForModel, {
            max_new_tokens: 512,
            do_sample: true,
            temperature: 0.7,
            top_k: 50,
            streamer: streamer, // <<< Pass the streamer instance
        });

        console.log(">>> DEBUG: await generator() call finished.");
        // Final state update is now handled reliably by streamer.end() -> updateUICallback

    } catch (error) {
        console.error(">>> DEBUG: Generation failed:", error);
        updateStatus(`Error generating response: ${error.message || error}`, true);
        if (assistantContentSpan) assistantContentSpan.textContent = `Error: ${error.message || error}`;
        else addMessageToChat('error', `Generation failed: ${error.message || error}`);
        if(assistantMsgDiv) {assistantMsgDiv.classList.add('error'); assistantMsgDiv.classList.remove('assistant');}
        isGenerating = false;
        setGeneratingState(false);
    }
}


// --- State Management Functions ---
// (Keep setLoadingState and setGeneratingState as they were, including logs)
function setLoadingState(loading) {
    loadModelButton.disabled = loading;
    modelNameInput.disabled = loading;
    sendMessageButton.disabled = loading;
    userMessageInput.disabled = loading;
    // console.log(`>>> DEBUG: setLoadingState(${loading}). Load controls disabled: ${loading}. Chat controls disabled: ${loading}.`);
}

function setGeneratingState(generating, modelReady = false) {
    const isModelLoaded = !!generator;
    const shouldChatBeDisabled = generating || isLoading || !isModelLoaded;
    const shouldLoadBeDisabled = generating || isLoading;

    sendMessageButton.disabled = shouldChatBeDisabled;
    userMessageInput.disabled = shouldChatBeDisabled;
    loadModelButton.disabled = shouldLoadBeDisabled;
    modelNameInput.disabled = shouldLoadBeDisabled;

    // console.log(`>>> DEBUG: setGeneratingState(generating=${generating}, modelReady=${modelReady}). Final state -> Chat Disabled: ${shouldChatBeDisabled}, Load Disabled: ${shouldLoadBeDisabled}. (isLoading=${isLoading}, isModelLoaded=${isModelLoaded})`);
}


// --- Event Listeners ---
// (Keep Event Listeners as they were)
loadModelButton.addEventListener('click', loadModel);
modelNameInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); loadModel(); }
});
sendMessageButton.addEventListener('click', generateResponse);
userMessageInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); generateResponse(); }
});

// --- Initial State ---
// (Keep Initial State setup as it was)
updateStatus("Page loaded. Ready to load a model.");
setLoadingState(false);
setGeneratingState(false);
console.log(">>> DEBUG: Initial page setup complete.");
