function formatCode(_imgData) {
	let format = ""
	let cellCount = 0 
	let emptyCount = 0
	for (let i = 0; i <= 4 * sizes.full.width * sizes.full.height; i += 1) {
		if(code.preformated.length > cellCount) {
			if((i+1)%4 == 0) {
				if(_imgData.data[i] == 0) {
					format += "\xa0"
					emptyCount++
				} else {
					format += code.preformated[cellCount]
					cellCount++
				}

			}
			if((i+1)%(4*sizes.full.width) == 0){

				format += "\n"
			}
		} else {
			code.formated = format
		}
	}
	code.formated = format
	setFormatedCode()

}

function postFormatCode(){
	code.postFormated = code.formated.replace("   ", "\t");
}

function setFormatedCode() {
	codeTextarea.value = code.formated
}