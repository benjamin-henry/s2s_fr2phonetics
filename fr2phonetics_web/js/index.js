const input_token_index = {" ": 0, "'": 1, ",": 2, "-": 3, ".": 4, "a": 5, "b": 6, "c": 7, "d": 8, "e": 9, "f": 10, "g": 11, "h": 12, "i": 13, "j": 14, "k": 15, "l": 16, "m": 17, "n": 18, "o": 19, "p": 20, "q": 21, "r": 22, "s": 23, "t": 24, "u": 25, "v": 26, "w": 27, "x": 28, "y": 29, "z": 30, "\u00e0": 31, "\u00e1": 32, "\u00e2": 33, "\u00e4": 34, "\u00e6": 35, "\u00e7": 36, "\u00e8": 37, "\u00e9": 38, "\u00ea": 39, "\u00eb": 40, "\u00ee": 41, "\u00ef": 42, "\u00f1": 43, "\u00f4": 44, "\u00f6": 45, "\u00fb": 46, "\u00fc": 47, "\u0153": 48};
const target_token_index = {"\t": 0, "\n": 1, " ": 2, "(": 3, ")": 4, ".": 5, "a": 6, "b": 7, "d": 8, "e": 9, "f": 10, "h": 11, "i": 12, "j": 13, "k": 14, "l": 15, "m": 16, "n": 17, "o": 18, "p": 19, "s": 20, "t": 21, "u": 22, "v": 23, "w": 24, "y": 25, "z": 26, "\u014b": 27, "\u0153": 28, "\u0251": 29, "\u0254": 30, "\u0259": 31, "\u025b": 32, "\u0261": 33, "\u0265": 34, "\u0272": 35, "\u0281": 36, "\u0283": 37, "\u0292": 38, "\u0294": 39, "\u02d0": 40, "\u0303": 41, "\u203f": 42};

const max_encoder_seq_length = 75;
const max_decoder_seq_length = 75;
const num_encoder_tokens = Object.keys(input_token_index).length;
const num_decoder_tokens = Object.keys(target_token_index).length;

let reversed_target_char_index = {};
for (let key in target_token_index) {
    reversed_target_char_index[target_token_index[key]]=key;
}

const encoderJson = 'http://localhost/benjamin/Perso/fr2phonetics_web/js/encoder/model.json';
const decoderJson = 'http://localhost/benjamin/Perso/fr2phonetics_web/js/decoder/model.json';

let encoder_model;
let decoder_model;

let translating = false;

$(document).ready(
    function() {
        loadModel();
        $("#clean").click(
            function() {
                document.getElementById('input_text').value = "";
                document.getElementById('translation_field').innerHTML = "";
            }
        )
        $("#translate").click(
            function() {
                translating = true;
                translate();
                translating = false;
            }
        )
    }
)

function translate() {
    let input_sequence = encodeWord(document.getElementById('input_text').value.toLowerCase())
    let data = tf.tensor([input_sequence]);
    let states_value = encoder_model.predict(data);

    let target_seq = new Array(num_decoder_tokens).fill(0.);
    target_seq = tf.tensor([[target_seq]]);
    decoder_input_data = [target_seq,states_value[0], states_value[1]];

    let stop_condition = false;
    let decoded_sentence = "";

    while(!stop_condition) {
        let decoder_output = decoder_model.predict(decoder_input_data);
        let sampled_token_index = decoder_output[0].argMax(axis=-1).dataSync();
        let sampled_char = reversed_target_char_index[sampled_token_index];
        decoded_sentence += sampled_char;
        if(sampled_char == "\n" || decoded_sentence.length > max_decoder_seq_length) {
            stop_condition = true;
        }
        target_seq = new Array(num_decoder_tokens).fill(0.);
        target_seq[sampled_token_index] = 1.;
        decoder_input_data = [tf.tensor([[target_seq]]),decoder_output[1], decoder_output[2]];
    }
    document.getElementById("translation_field").innerHTML = decoded_sentence;
}

async function loadModel() {
    try {
        encoder_model = await tf.loadLayersModel(encoderJson);
        // encoder_model.summary();
        console.log("[INFO] Encoder loaded");
        decoder_model = await tf.loadLayersModel(decoderJson);
        // decoder_model.summary();
        console.log("[INFO] Decoder loaded");
    } catch (error) {
        console.log(error);
    }   
}

function encodeWord(word) {
    let seq = [];
    for (let i = 0; i < word.length;i++) {
      let temp = new Array(num_encoder_tokens).fill(0.);
      temp[input_token_index[word[i]]] = 1.;
      seq.push(temp);
    }
    do {
      let temp = new Array(num_encoder_tokens).fill(0.);
      temp[input_token_index[" "]] = 1.;
      seq.push(temp);
    }while(seq.length < max_encoder_seq_length);
    return seq; 
}






