const codeTextarea = document.getElementById("codeTextarea")
codeTextarea.addEventListener("change", handleCodeChange)
codeTextarea.addEventListener("focusin", handleFocuseIn);
codeTextarea.addEventListener("focusout", handleFocuseOut);

const fabFormatInput = document.getElementById("fabFormatInput")
fabFormatInput.addEventListener("click", handleSync)

const sizes = {
	single: {
		width: 0,
		height: 0
	},
	full: {
		width: 0,
		height: 0
	}
}

const code = {
	pretty: {
		sync: false,
		paused: false
	}
}
let video

/**
 * Checking character sizes to know what 1 pixel size should be
 */

function setSizes() {
	const testCharSize = document.getElementById("testCharSize")
	
	sizes.single.width = testCharSize.clientWidth/10
	sizes.single.height = testCharSize.clientHeight/10

	sizes.full.width = Math.floor(window.innerWidth/sizes.single.width)
	sizes.full.height = Math.floor(window.innerHeight/sizes.single.height)

	return sizes
}
/**
 * Getting the code from the text area and removing all the breaklines and tabs.
 * This happens when the textarea's value changes.
 */



function handleCodeChange() {
	code.original = codeTextarea.value
	code.preformated = preformat(codeTextarea.value)
}

function testChange(){
	const testValue = `// Copyright (c) 2018 ml5\n//\n// This software is released under the MIT License.\n// https://opensource.org/licenses/MIT\n\n/*\nImage Classifier using pre-trained networks\n*/\n\nimport * as tf from '@tensorflow/tfjs';\nimport callCallback from '../utils/callcallback';\nimport generatedImageResult from '../utils/generatedImageResult';\nimport handleArguments from "../utils/handleArguments";\nimport { mediaReady } from '../utils/imageUtilities';\n\nconst DEFAULTS = {\n  modelPath: 'https://raw.githubusercontent.com/zaidalyafeai/HostedModels/master/unet-128/model.json',\n  imageSize: 128,\n  returnTensors: false,\n}\n\nclass UNET {\n  /**\n   * Create UNET class. \n   * @param {HTMLVideoElement | HTMLImageElement} video - The video or image to be used for segmentation.\n   * @param {Object} options - Optional. A set of options.\n   * @param {function} callback - Optional. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise \n   *    that will be resolved once the model has loaded.\n   */\n  constructor(video, options, callback) {\n    this.modelReady = false;\n    this.isPredicting = false;\n    this.config = {\n      modelPath: typeof options.modelPath !== 'undefined' ? options.modelPath : DEFAULTS.modelPath,\n      imageSize: typeof options.imageSize !== 'undefined' ? options.imageSize : DEFAULTS.imageSize,\n      returnTensors: typeof options.returnTensors !== 'undefined' ? options.returnTensors : DEFAULTS.returnTensors,\n    };\n    this.video = video;\n    this.ready = callCallback(this.loadModel(), callback);\n  }\n\n  async loadModel() {\n    this.model = await tf.loadLayersModel(this.config.modelPath);\n    this.modelReady = true;\n    return this;\n  }\n\n  async segment(inputOrCallback, cb) {\n    const { image, callback } = handleArguments(this.video, inputOrCallback, cb);\n    return callCallback(this.segmentInternal(image), callback);\n  }\n\n  async segmentInternal(imgToPredict) {\n    // Wait for the model to be ready and video input to be loaded\n    await this.ready;\n    await mediaReady(imgToPredict, true);\n    this.isPredicting = true;\n\n    const {\n      featureMask,\n      backgroundMask,\n      segmentation\n    } = tf.tidy(() => {\n      // preprocess the input image\n      const tfImage = tf.browser.fromPixels(imgToPredict).toFloat();\n      const resizedImg = tf.image.resizeBilinear(tfImage, [this.config.imageSize, this.config.imageSize]);\n      let normTensor = resizedImg.div(tf.scalar(255));\n      const batchedImage = normTensor.expandDims(0);\n      // get the segmentation\n      const pred = this.model.predict(batchedImage);\n      \n      // add back the alpha channel to the normalized input image\n      const alpha = tf.ones([128, 128, 1]).tile([1,1,1])\n      normTensor = normTensor.concat(alpha, 2)\n\n      // TODO: optimize these redundancies below, e.g. repetitive squeeze() etc\n      // get the background mask;\n      let maskBackgroundInternal = pred.squeeze([0]);\n      maskBackgroundInternal = maskBackgroundInternal.tile([1, 1, 4]);\n      maskBackgroundInternal = maskBackgroundInternal.sub(0.3).sign().relu().neg().add(1);\n      const featureMaskInternal = maskBackgroundInternal.mul(normTensor);\n\n      // get the feature mask;\n      let maskFeature = pred.squeeze([0]);\n      maskFeature = maskFeature.tile([1, 1, 4]);\n      maskFeature = maskFeature.sub(0.3).sign().relu();\n      const backgroundMaskInternal = maskFeature.mul(normTensor);\n\n      const alpha255 = tf.ones([128, 128, 1]).tile([1,1,1]).mul(255);\n      let newpred = pred.squeeze([0]);\n      newpred = tf.cast(newpred.tile([1,1,3]).sub(0.3).sign().relu().mul(255), 'int32') \n      newpred = newpred.concat(alpha255, 2)\n\n      return {\n        featureMask: featureMaskInternal,\n        backgroundMask: backgroundMaskInternal,\n        segmentation: newpred\n      };\n    });\n\n    this.isPredicting = false;\n\n    const maskFeat = await generatedImageResult(featureMask, this.config);\n    const maskBg = await generatedImageResult(backgroundMask, this.config);\n    const mask = await generatedImageResult(segmentation, this.config);\n\n    return {\n      segmentation: mask.raw,\n      blob: {\n        featureMask: maskFeat.blob,\n        backgroundMask: maskBg.blob\n      },\n      tensor: {\n        featureMask: maskFeat.tensor,\n        backgroundMask: maskBg.tensor,\n      },\n      raw: {\n        featureMask: maskFeat.raw,\n        backgroundMask: maskBg.raw\n      },\n      // returns if p5 is available\n      featureMask: maskFeat.image,\n      backgroundMask: maskBg.image,\n      mask: mask.image\n    };\n  }\n}\n\nconst uNet = (...inputs) => {\n  const { video, options = {}, callback } = handleArguments(...inputs);\n  return new UNET(video, options, callback);\n};\n\nexport default uNet;`;
	codeTextarea.value = testValue
	handleCodeChange()
}

function preformat(input) {
	input = input.replace(/\s{2,}/g, ' ');
	input = input.replace(/\t/g, ' ');
	input = input.toString().trim().replace(/(\r\n|\n|\r)/g,"");
	return input
}

/**
 * 
 */

async function setStream() {
	setSizes()
	let stream = null;
	const constraints = {
		audio: false,
		video: {
			frameRate: { max: 12 },
		  	width: { min: sizes.full.width },
		  	height: { min: sizes.full.height },
		}
	}
	try {
		stream = await navigator.mediaDevices.getUserMedia(constraints);
		video = setVideo(stream)
	  } catch (err) {
		/* handle the error */
	  }
}

function setVideo(stream) {
	const myVideo = document.createElement("video")
	myVideo.width = sizes.full.width
	myVideo.height = sizes.full.height
	myVideo.srcObject = stream
	myVideo.onloadedmetadata = () => {
		myVideo.play();
		video = myVideo
		return myVideo
	};

}

setStream()
testChange()

function handleSync() {
	if(code.pretty.sync){
		code.pretty.sync = false
	} else {
		code.pretty.sync = true
		setupMl()
	}
}
function handleFocuseIn(){
	if(code.pretty.sync){
		code.pretty.sync = false
		code.pretty.paused = true
	}
}
function handleFocuseOut(){
	if(code.pretty.paused){
		code.pretty.sync = true
		code.pretty.paused = false
		setupMl()
	}
}