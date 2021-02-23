video = $('#webcam')[0];
const canvas = $('#overlay')[0];
const ctx = canvas.getContext('2d');

const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;

let model;

var hm = document.querySelector('.heatmap');

var heatmap = h337.create({
    container: hm,
    radius: 60
});

$('#clrhm').click(function() {
    heatmap.setData({ data: [] })
});


var trackData = false;

setInterval(function() {
    trackData = true;
}, 100);

var idleTimeout, idleInterval;

var lastX, lastY;
var idleCount;

function startIdle() {
    idleCount = 0;

    function idle() {
        heatmap.addData({
            x: lastX,
            y: lastY
        });
        idleCount++;
        if (idleCount > 10) {
            clearInterval(idleInterval);
        }
    };
    idle();
    idleInterval = setInterval(idle, 1000);
};


/* $(document).mousemove(function(e) {
    const $target = $('#target');
    $target.css('left', e.pageX + 'px');
    $target.css('top', e.pageY + 'px');
    // Add data to the heatmap
    if (idleTimeout) clearTimeout(idleTimeout);
    if (idleInterval) clearInterval(idleInterval);

    if (trackData) {
        lastX = e.pageX;
        lastY = e.pageY;
        heatmap.addData({
            x: lastX,
            y: lastY
        });
        trackData = false;
    }
    idleTimeout = setTimeout(startIdle, 500);
}); */



async function setupCamera() {

    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            // Only setting the video to a specified size in order to accommodate a
            // point cloud, so on mobile devices accept the default size.
            width: 400,
            height: 300
        },
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

function distance(a, b) {
    return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

async function pred() {
    // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain an
    // array of detected faces from the MediaPipe graph. If passing in a video
    // stream, a single prediction per frame will be returned.
    const predictions = await model.estimateFaces({
        input: video,
    });

    // ctx.clearRect(0, 0, canvas.width,canvas.height);
    ctx.drawImage(
        video, 0, 0, canvas.width, canvas.height);


    if (predictions.length > 0) {
        for (let i = 0; i < predictions.length; i++) {
            const keypoints = predictions[i].scaledMesh;

            // Log facial keypoints.
            for (let i = 0; i < keypoints.length; i++) {
                const x = keypoints[i][0];
                const y = keypoints[i][1];

                ctx.beginPath();
                ctx.arc(x, y, 1, 0, 2 * Math.PI);
                ctx.fillStyle = "aqua";
                if (i == 111 || i == 46 || i == 300 || i == 276)
                    ctx.fillStyle = "red";
                ctx.fill();
            }

            ctx.strokeStyle = "red";
            ctx.lineWidth = 1;
            const leftCenter = keypoints[NUM_KEYPOINTS];
            const leftDiameterY = distance(
                keypoints[NUM_KEYPOINTS + 4],
                keypoints[NUM_KEYPOINTS + 2]);
            const leftDiameterX = distance(
                keypoints[NUM_KEYPOINTS + 3],
                keypoints[NUM_KEYPOINTS + 1]);

            ctx.beginPath();
            ctx.ellipse(leftCenter[0], leftCenter[1], leftDiameterX / 2, leftDiameterY / 2, 0, 0, 2 * Math.PI);
            ctx.stroke();

            const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
            const rightDiameterY = distance(
                keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
                keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]);
            const rightDiameterX = distance(
                keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
                keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]);


            ctx.beginPath();
            ctx.ellipse(rightCenter[0], rightCenter[1], rightDiameterX / 2, rightDiameterY / 2, 0, 0, 2 * Math.PI);
            ctx.stroke();

            var eyex = keypoints[111][0],
                eyey = keypoints[111][1],
                eyeh = (keypoints[46][1] - keypoints[111][1]),
                eyew = (keypoints[276][0] - keypoints[111][0]);


            ctx.strokeRect(eyex, eyey, eyew, eyeh);



            // The video might internally have a different size, so we need these
            // factors to rescale the eyes rectangle before cropping:
            const resizeFactorX = video.videoWidth / video.width;
            const resizeFactorY = video.videoHeight / video.height;

            // // Crop the eyes from the video and paste them in the eyes canvas:
            const eyesCanvas = $('#eyes')[0];
            const eyesCC = eyesCanvas.getContext('2d');

            eyesCC.drawImage(
                video,
                eyex * resizeFactorX, eyey * resizeFactorY,
                eyew * resizeFactorX, eyeh * resizeFactorY,
                0, 0, eyesCanvas.width, eyesCanvas.height
            );
        }
    }
    // ctx.restore();
    var rafID = requestAnimationFrame(pred);
}

async function main() {
    await tf.setBackend('webgl');
    await setupCamera();
    // video.play();

    model = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh);

    requestAnimationFrame(pred);
}


main();

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

var dataset = {
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
    // console.log(dataset)
    upd();

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
                $('#status').text("Epochs: " + n + '/20');
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

        const targetWidth = $('#target').outerWidth();
        const targetHeight = $('#target').outerHeight();

        prediction.data().then(prediction => {
            const x = ((prediction[0] + 1) / 2) * ($(window).width() - targetWidth);
            const y = ((prediction[1] + 1) / 2) * ($(window).height() - targetHeight);

            // Move target there:
            const $target = $('#target');
            $target.css('left', x + 'px');
            $target.css('top', y + 'px');

            // Add data to the heatmap
            if (idleTimeout) clearTimeout(idleTimeout);
            if (idleInterval) clearInterval(idleInterval);

            if (trackData) {
                lastX = x;
                lastY = y;
                heatmap.addData({
                    x: lastX,
                    y: lastY
                });
                trackData = false;
            }
            idleTimeout = setTimeout(startIdle, 500);
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
    document.getElementById(name).style.display = "block ";
    document.getElementById(name).className += ' active '

}

var is_colliding = function($div1, $div2) {
    // Div 1 data
    var d1_offset = $div1.offset();
    var d1_height = $div1.outerHeight(true);
    var d1_width = $div1.outerWidth(true);
    var d1_distance_from_top = d1_offset.top + d1_height;
    var d1_distance_from_left = d1_offset.left + d1_width;

    // Div 2 data
    var d2_offset = $div2.offset();
    var d2_height = $div2.outerHeight(true);
    var d2_width = $div2.outerWidth(true);
    var d2_distance_from_top = d2_offset.top + d2_height;
    var d2_distance_from_left = d2_offset.left + d2_width;

    var not_colliding = (d1_distance_from_top < d2_offset.top || d1_offset.top > d2_distance_from_top || d1_distance_from_left < d2_offset.left || d1_offset.left > d2_distance_from_left);

    // Return whether it IS colliding
    return !not_colliding;
};

function download(content, fileName, contentType) {
    const a = document.createElement('a');
    const file = new Blob([content], {
        type: contentType,
    });
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
}

$('#down').click(function() {
    const data = toJSON();
    const json = JSON.stringify(data);
    download(json, 'dataset.json', 'text/plain');
});

$('#clr').click(function() {
    dataset = {
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
    };
    heatmap.setData({ data: [] })
    upd();
});

var tag = document.createElement('script');

tag.src = "https://www.youtube.com/iframe_api";
var firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

var player;
var vId = '8BdSzXQ6fCU';

$('#play').click(function() {
    try {
        player.loadVideoById($('#vid').val());
    } catch (error) {
        alert(error)
    }
    heatmap.setData({ data: [] })
});

function onYouTubeIframeAPIReady() {
    player = new YT.Player('drive', {
        height: '472.5',
        width: '840',
        videoId: vId,
        playerVars: {
            start: 441,
            controls: 0,
        },
        events: {
            'onReady': onPlayerReady,
            'onStateChange': onPlayerStateChange
        }
    });
}

function onPlayerReady(event) {
    // event.target.playVideo();
}

var snd = new Audio("sound.mp3");

function show() {
    snd.play();
    $('#text').css('display', 'inline');
    setTimeout(hide, 5000); // 5 seconds
}

function hide() {
    $('#text').css('display', 'none');
}

function onPlayerStateChange(event) {
    if (event.data == YT.PlayerState.PLAYING) {
        var t = 0,
            f = 0;
        console.log('Playing')
        var r1 = Math.random() * 26
        var r2 = Math.random() * 56 + 30

        console.log(r1, r2)
        setTimeout(() => {
            show()
        }, r1 * 1000)
        setTimeout(() => {
            show()
        }, r2 * 1000)
        window.setInterval(function() {
            // console.log(is_colliding($('#drive'), $('#target')));
            if (is_colliding($('#drive'), $('#target'))) {
                $('#drive').css('border', '5px solid green');
                t++;
            } else {
                $('#drive').css('border', '5px solid red');
                f++;
            }
        }, 500);
        setTimeout(() => {
            window.clearInterval();
            var p = t / (t + f) * 100;
            $('#drive').css('border', 'none');
            stopVideo()
            alert(p + '%')
        }, 60000);
        // done = true;
    }
}

function stopVideo() {
    player.stopVideo();
}

document.getElementById('contentFile').onchange = function(e) {
    const file = e.target.files[0];
    const reader = new FileReader();

    reader.onload = function() {
        const data = reader.result;
        const json = JSON.parse(data);
        fromJSON(json);
        upd();
    };

    reader.readAsBinaryString(file);
}

function upd() {
    $('#nt').text("Train: " + dataset['train']['n']);
    $('#nv').text("Validation: " + dataset['val']['n']);
}

function toJSON() {
    const tensorToArray = function(t) {
        const typedArray = t.dataSync();
        return Array.prototype.slice.call(typedArray);
    };

    return {
        train: {
            shapes: {
                x: dataset.train.x.shape,
                y: dataset.train.y.shape,
            },
            n: dataset.train.n,
            x: tensorToArray(dataset.train.x),

            y: tensorToArray(dataset.train.y),
        },
        val: {
            shapes: {
                x: dataset.val.x.shape,
                y: dataset.val.y.shape,
            },
            n: dataset.val.n,
            x: tensorToArray(dataset.val.x),

            y: tensorToArray(dataset.val.y),
        },
    };
};

function fromJSON(data) {
    dataset.train.n = data.train.n;
    dataset.train.x = data.train.x && [
        tf.tensor(data.train.x, data.train.shapes.x),
    ];
    dataset.train.y = tf.tensor(data.train.y, data.train.shapes.y);
    dataset.val.n = data.val.n;
    dataset.val.x = data.val.x && [
        tf.tensor(data.val.x, data.val.shapes.x),
    ];
    dataset.val.y = tf.tensor(data.val.y, data.val.shapes.y);
};