// Audio Detection JS
const audioForm = document.getElementById('audio-upload-form');
const audioFileInput = document.getElementById('audio-file');
const audioPreviewSection = document.getElementById('audio-preview-section');
const audioPreview = document.getElementById('audio-preview');
const audioLoading = document.getElementById('audio-loading');
const audioResults = document.getElementById('audio-results');
const audioPredictionResult = document.getElementById('audio-prediction-result');
const audioPredictionBars = document.getElementById('audio-prediction-bars');
const audioSpectrograms = document.getElementById('audio-spectrograms');
const audioError = document.getElementById('audio-error');

if (audioFileInput) {
    audioFileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            audioPreview.src = URL.createObjectURL(file);
            audioPreviewSection.classList.remove('d-none');
        } else {
            audioPreviewSection.classList.add('d-none');
        }
    });
}

if (audioForm) {
    audioForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        audioError.classList.add('d-none');
        audioResults.classList.add('d-none');
        audioLoading.classList.remove('d-none');
        const file = audioFileInput.files[0];
        if (!file) {
            audioError.textContent = 'Please select an audio file.';
            audioError.classList.remove('d-none');
            audioLoading.classList.add('d-none');
            return;
        }
        const formData = new FormData();
        formData.append('audio', file);
        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            audioLoading.classList.add('d-none');
            if (data.error) {
                audioError.textContent = data.error;
                audioError.classList.remove('d-none');
                return;
            }
            // Show results
            audioResults.classList.remove('d-none');
            // Prediction result
            if (data.status === 'confident_match') {
                audioPredictionResult.innerHTML = `<strong>Predicted Bird:</strong> <span class='text-primary'>${data.predicted_label}</span><br><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%`;
            } else if (data.status === 'close_match') {
                audioPredictionResult.innerHTML = `<strong>Close to:</strong> <span class='text-primary'>${data.predicted_label}</span><br><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%`;
            } else {
                audioPredictionResult.innerHTML = `<strong>No Target Bird Call Detected</strong><br>Top Matches:<br>` +
                    data.top_matches.map(match => `• ${match.label} (${(match.confidence * 100).toFixed(2)}%)`).join('<br>');
            }
            // Prediction bars
            audioPredictionBars.innerHTML = '';
            Object.entries(data.predictions).sort((a, b) => b[1] - a[1]).forEach(([label, prob]) => {
                const percentage = (prob * 100).toFixed(2);
                audioPredictionBars.innerHTML += `<div class='mb-2'><div class='d-flex justify-content-between'><span>${label}</span><span>${percentage}%</span></div><div class='progress'><div class='progress-bar bg-primary' role='progressbar' style='width: ${percentage}%' aria-valuenow='${percentage}' aria-valuemin='0' aria-valuemax='100'></div></div></div>`;
            });
            // Spectrograms
            audioSpectrograms.innerHTML = '';
            if (data.mel_spectrograms && data.mel_spectrograms.length > 0) {
                audioSpectrograms.innerHTML = '<h6>Mel Spectrograms</h6>';
                data.mel_spectrograms.forEach((spec, idx) => {
                    audioSpectrograms.innerHTML += `<img src='data:image/png;base64,${spec}' alt='Spectrogram ${idx+1}' class='img-fluid rounded mb-2' style='max-width:150px;'>`;
                });
            }
        } catch (err) {
            audioLoading.classList.add('d-none');
            audioError.textContent = 'An error occurred while processing the audio.';
            audioError.classList.remove('d-none');
        }
    });
    audioForm.addEventListener('reset', function() {
        audioPreview.src = '';
        audioPreviewSection.classList.add('d-none');
        audioResults.classList.add('d-none');
        audioError.classList.add('d-none');
    });
} 