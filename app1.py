from flask import Flask, render_template, request
import pickle
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io

from PIL import Image
import PIL
app= Flask(__name__,static_url_path="/static/")
@app.route('/')
def man():
  return render_template('home_trial.html')


@app.route('/predict', methods=['POST'])
def home():
    if request.method == 'POST':
        img=request.files["img"]
        imge=img.read()
        u=Image.open(io.BytesIO(imge))
        u = u.resize((244, 244), resample=PIL.Image.LANCZOS)

        img1 = np.array(u)
        
        image_batch = np.expand_dims(img1, axis=0)
       





        if img is None:
            x=["I","am","love"]
        else:
            from tensorflow import keras
            img_model=keras.models.load_model('VGG16final')
            decoder_model=keras.models.load_model('decoder_model_final')

            with open('tokenizer.pickle', 'rb') as handle:
                tokeniser = pickle.load(handle)
            transfered_values = img_model.predict(image_batch)
            
            start_word = tokeniser.word_index['sos']
            end_word = tokeniser.word_index['eos']
            corpus_index = tokeniser.word_index
            corpus_index = {value: key for key, value in corpus_index.items()}
            decode_input = np.zeros(shape=(1, 10), dtype=np.int)

            curr_token = start_word
            count_tokens = 0
            max_token = 10
            output_text = []
            while curr_token != end_word and count_tokens < max_token:
                decode_input[0, count_tokens] = curr_token
                x_data = {
                    'transfer_values_input': transfered_values,
                    'decoders_input': decode_input
                }
                decode_output = decoder_model.predict(x_data)
                token_output = decode_output[0, count_tokens, :]
                pred = np.argmax(token_output)

                curr_token = pred
                
                sampled_word = corpus_index[pred]
                output_text.append(sampled_word)
                count_tokens += 1

            x=output_text
        return render_template('second-page.html', a=x)
    return render_template('home_trial.html')




@app.route('/sender', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sent_word=request.form.getlist('mycheckbox')
        input_word = 'sos'
        for ele in sent_word:
            input_word=input_word+' '+ele
        ip_word=input_word[3:]
        total_words = 10000
        max_sequence_len=171
        model = keras.Sequential()
        model.add(keras.layers.Embedding(total_words, 100, input_length=max_sequence_len - 1))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.TimeDistributed(keras.layers.Dense(128, activation='softmax')))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(total_words, activation='softmax'))
        nadam = keras.optimizers.Nadam(lr=0.001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
        model.load_weights('transformer_weights_128_3')
        with open('tokenizer_for_poem.pickle', 'rb') as handle:
            tokeniser = pickle.load(handle)
        corpus = tokeniser.word_index
        reverse_corpus = {value: key for key, value in corpus.items()}
        token = tokeniser.texts_to_sequences([input_word])
        padded = np.array(pad_sequences(token, maxlen=max_sequence_len - 1))
        count = 0
        while count < 7:
            output_token = model.predict(padded)
            output_token = np.argmax(output_token)
            word_pred = reverse_corpus[output_token]
            input_word = input_word + " " + word_pred
            if word_pred == 'sos':
                ip_word = ip_word + "\n"
                count += 1
            elif word_pred != 'eos':
                ip_word = ip_word + " " + word_pred

            token = tokeniser.texts_to_sequences([input_word])
            padded = np.array(pad_sequences(token, maxlen=max_sequence_len - 1))
        for i,ele in enumerate(ip_word):
            if ele == " ":
                ip_word=ip_word[i+1:]
            else:
                break
        return render_template('result_final.html', data=ip_word)


    return render_template('home_trial.html')

if __name__ == '__main__':
    app.run(debug=True)
