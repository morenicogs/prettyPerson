const uNet = ml5.uNet("face");

function setupMl() {
	uNet.segment(video, gotResult)
}

function gotResult(error, result) {
	personCanvas(result)
}

function personCanvas(segImg) {
	if (Object.prototype.hasOwnProperty.call(segImg, "raw")) {
		const uNetImg = imgData2Canvas(segImg.raw.backgroundMask, 128, 128)
		const offscreen = new OffscreenCanvas(sizes.full.width, sizes.full.height);
		const offContext = offscreen.getContext("2d")
		offContext.drawImage(uNetImg, 0, 0, 128, 128, 0, 0, sizes.full.width, sizes.full.height)
		setTimeout(() => {
			// uNet.segment(video, gotResult);
			// request = requestAnimationFrame(draw);
			const newImageDat = offContext.getImageData(0,0,sizes.full.width,sizes.full.height)
		
			if(code.pretty.sync){
				formatCode(newImageDat)
				uNet.segment(video, gotResult)
			}
		}, 42);
	}
}

function imgData2Canvas(imageData, w, h) {
	const imgData = new ImageData(imageData, w)
	const offscreen = new OffscreenCanvas(w, h);
	const offContext = offscreen.getContext("2d")
	offContext.putImageData(imgData, 0, 0);
	return offContext.canvas;
}