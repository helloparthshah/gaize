$(document).ready(function() {
    const video = $('#webcam')[0];

    function onStreaming(stream) {
        video.srcObject = stream;
    }

    navigator.mediaDevices.getUserMedia({ video: true }).then(onStreaming);
});