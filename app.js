$(document).ready(function() {
    const video = $('#webcam')[0];
    const ctrack = new clm.tracker();
    ctrack.init();

    const overlay = $('#overlay')[0];
    const overlayCC = overlay.getContext('2d');

    function trackingLoop() {

        requestAnimationFrame(trackingLoop);

        let currentPosition = ctrack.getCurrentPosition();
        overlayCC.clearRect(0, 0, 400, 300);

        if (currentPosition) {
            ctrack.draw(overlay);
        }

        if (currentPosition) {
            // Draw facial mask on overlay canvas:
            ctrack.draw(overlay);

            // Get the eyes rectangle and draw it in red:
            const eyesRect = getEyesRectangle(currentPosition);
            overlayCC.strokeStyle = 'red';
            overlayCC.strokeRect(eyesRect[0], eyesRect[1], eyesRect[2], eyesRect[3]);

            // The video might internally have a different size, so we need these
            // factors to rescale the eyes rectangle before cropping:
            const resizeFactorX = video.videoWidth / video.width;
            const resizeFactorY = video.videoHeight / video.height;

            // Crop the eyes from the video and paste them in the eyes canvas:
            const eyesCanvas = $('#eyes')[0];
            const eyesCC = eyesCanvas.getContext('2d');

            eyesCC.drawImage(
                video,
                eyesRect[0] * resizeFactorX, eyesRect[1] * resizeFactorY,
                eyesRect[2] * resizeFactorX, eyesRect[3] * resizeFactorY,
                0, 0, eyesCanvas.width, eyesCanvas.height
            );
        }
    }

    function onStreaming(stream) {
        video.srcObject = stream;
        ctrack.start(video);
        trackingLoop()
    }

    navigator.mediaDevices.getUserMedia({ video: true }).then(onStreaming);
});

function getEyesRectangle(positions) {
    const minX = positions[23][0] - 5;
    const maxX = positions[28][0] + 5;
    const minY = positions[24][1] - 5;
    const maxY = positions[26][1] + 5;

    const width = maxX - minX;
    const height = maxY - minY;

    return [minX, minY, width, height];
}

function getImage() {
    // Capture the current image in the eyes canvas as a tensor.
    return tf.tidy(function() {
        const image = tf.browser.fromPixels($('#eyes')[0]);
        // Add a batch dimension:
        const batchedImage = image.expandDims(0);
        // Normalize and return it:
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
}

let dataset = {
    train: {
        n: 0,
        x: null,
        y: null,
    },
    val: {
        n: 0,
        x: null,
        y: null,
    },
}

function captureExample() {
    // Take the latest image from the eyes canvas and add it to our dataset.
    tf.tidy(function() {
        const image = getImage();
        const mousePos = tf.tensor1d([mouse.x, mouse.y]).expandDims(0);

        // Choose whether to add it to training (80%) or validation (20%) set:
        const subset = dataset[Math.random() > 0.2 ? 'train' : 'val'];

        if (subset.x == null) {
            // Create new tensors
            subset.x = tf.keep(image);
            subset.y = tf.keep(mousePos);
        } else {
            // Concatenate it to existing tensors
            const oldX = subset.x;
            const oldY = subset.y;

            subset.x = tf.keep(oldX.concat(image, 0));
            subset.y = tf.keep(oldY.concat(mousePos, 0));
        }

        // Increase counter
        subset.n += 1;
    });
    console.log(dataset)

    $('#nt').text(dataset['train']['n']);
    $('#nv').text(dataset['val']['n']);

}

$('body').keyup(function(event) {
    // On space key:
    if (event.keyCode == 32) {
        captureExample();

        event.preventDefault();
        return false;
    }
});

// Track mouse movement:
const mouse = {
    x: 0,
    y: 0,

    handleMouseMove: function(event) {
        // Get the mouse position and normalize it to [-1, 1]
        mouse.x = (event.clientX / $(window).width()) * 2 - 1;
        mouse.y = (event.clientY / $(window).height()) * 2 - 1;
    },
}

document.onmousemove = mouse.handleMouseMove;

let currentModel;

function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 20,
        strides: 1,
        activation: 'relu',
        inputShape: [$('#eyes').height(), $('#eyes').width(), 3],
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
    }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dropout(0.2));

    // Two output values x and y
    model.add(tf.layers.dense({
        units: 2,
        activation: 'tanh',
    }));

    // Use ADAM optimizer with learning rate of 0.0005 and MSE loss
    model.compile({
        optimizer: tf.train.adam(0.0005),
        loss: 'meanSquaredError',
    });

    return model;
}

function fitModel() {
    let batchSize = Math.floor(dataset.train.n * 0.1);
    if (batchSize < 4) {
        batchSize = 4;
    } else if (batchSize > 64) {
        batchSize = 64;
    }

    if (currentModel == null) {
        currentModel = createModel();
    }
    let n = 0;
    currentModel.fit(dataset.train.x, dataset.train.y, {
        batchSize: batchSize,
        epochs: 20,
        shuffle: true,
        validationData: [dataset.val.x, dataset.val.y],
        callbacks: {
            onEpochEnd: async function(epoch, logs) {
                n++;
                $('#status').text(n + '/20');
            }
        }
    }).then(info => {
        alert('Done')
    });
}

$('#train').click(function() {
    fitModel();
});

function moveTarget() {
    if (currentModel == null) {
        return;
    }
    tf.tidy(function() {
        const image = getImage();
        const prediction = currentModel.predict(image);

        // Convert normalized position back to screen position:
        const targetWidth = $('#target').outerWidth();
        const targetHeight = $('#target').outerHeight();

        // It's okay to run this async, since we don't have to wait for it.
        prediction.data().then(prediction => {
            const x = ((prediction[0] + 1) / 2) * ($(window).width() - targetWidth);
            const y = ((prediction[1] + 1) / 2) * ($(window).height() - targetHeight);

            // Move target there:
            const $target = $('#target');
            $target.css('left', x + 'px');
            $target.css('top', y + 'px');
        });
    });
}

setInterval(moveTarget, 10);

openTab(event, 'Train');

function openTab(evt, name) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab ");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none ";
    }
    /* tablinks = document.getElementsByClassName("tablinks ");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active ", " ");
    } */
    document.getElementById(name).style.display = "block ";
    document.getElementById(name).className += ' active '

}

/* function downloadObjectAsJson(exportObj, exportName) {
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj));
    console.log(dataset)
    console.log(dataStr)
    var downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", exportName + ".json");
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
} */

function download(content, fileName) {
    const a = document.createElement('a');
    const file = new Blob([content], {
        type: 'data:text/json;charset=utf-8',
    });
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
}

$('#down').click(function() {
    download(dataset, 'dataset.json');
    // downloadObjectAsJson(dataset, 'dataset.json')
});

document.getElementById('contentFile').onchange = function(e) {
    const file = e.target.files[0];
    const reader = new FileReader();

    reader.onload = function() {
        const data = reader.result;
        const json = JSON.parse(data);
        dataset.fromJSON(json);
        $('#nt').text(dataset['train']['n']);
        $('#nv').text(dataset['val']['n']);
    };

    reader.readAsBinaryString(file);
}