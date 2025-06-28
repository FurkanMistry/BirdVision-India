// Image/Video Detection JS
const form = document.getElementById('image-video-upload-form');
const fileInput = document.getElementById('file-upload');
const resultContainer = document.getElementById('result-container');
const resultImage = document.getElementById('result-image');
const detectionsList = document.getElementById('detections-list');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('error-message');
const imageOption = document.getElementById('image-option');
const videoOption = document.getElementById('video-option');

let detectionType = 'image';

if (imageOption && videoOption) {
    imageOption.addEventListener('change', () => { detectionType = 'image'; });
    videoOption.addEventListener('change', () => { detectionType = 'video'; });
}

if (form) {
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        errorMessage.classList.add('d-none');
        resultContainer.classList.add('d-none');
        loading.classList.remove('d-none');
        const file = fileInput.files[0];
        if (!file) {
            errorMessage.textContent = 'Please select a file.';
            errorMessage.classList.remove('d-none');
            loading.classList.add('d-none');
            return;
        }
        const formData = new FormData();
        formData.append('file', file);
        try {
            let url = detectionType === 'image' ? '/process_image' : '/process_video';
            const response = await fetch(url, { method: 'POST', body: formData });
            const data = await response.json();
            loading.classList.add('d-none');
            if (data.error) {
                errorMessage.textContent = data.error;
                errorMessage.classList.remove('d-none');
                return;
            }
            if (detectionType === 'image' && data.result_path) {
                resultImage.src = data.result_path;
                resultImage.classList.remove('d-none');
                detectionsList.innerHTML = data.detections && data.detections.length > 0 ? data.detections.map(d => `<span class='badge bg-primary me-1 mb-1'>${d}</span>`).join('') : 'No birds detected.';
                resultContainer.classList.remove('d-none');
            } else if (detectionType === 'video') {
                detectionsList.innerHTML = 'Video processed. Check the camera window.';
                resultImage.classList.add('d-none');
                resultContainer.classList.remove('d-none');
            }
        } catch (err) {
            loading.classList.add('d-none');
            errorMessage.textContent = 'An error occurred while processing the file.';
            errorMessage.classList.remove('d-none');
        }
    });
    form.addEventListener('reset', function() {
        resultImage.src = '';
        resultContainer.classList.add('d-none');
        errorMessage.classList.add('d-none');
    });
} 