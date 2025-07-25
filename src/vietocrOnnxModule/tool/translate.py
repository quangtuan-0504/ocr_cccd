import torch
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.vocab import Vocab
from model.transformerocr import VietOCR
import math
from PIL import Image


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token] * len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = output.to('cpu')
            
            values, indices = torch.topk(output, 1)
            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence

def translate_onnx(img, session, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    cnn_session, encoder_session, decoder_session = session
    
    # create cnn input
    cnn_input = {cnn_session.get_inputs()[0].name: img}
    src = cnn_session.run(None, cnn_input)
    
    # create encoder input
    encoder_input = {encoder_session.get_inputs()[0].name: src[0]}
    encoder_outputs, hidden = encoder_session.run(None, encoder_input)
    translated_sentence = [[sos_token] * len(img)]
    max_length = 0

    while max_length <= max_seq_length and not all(
        np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
    ):
        tgt_inp = translated_sentence
        decoder_input = {decoder_session.get_inputs()[0].name: tgt_inp[-1], decoder_session.get_inputs()[1].name: hidden, decoder_session.get_inputs()[2].name: encoder_outputs}

        output, hidden, _ = decoder_session.run(None, decoder_input)
        output = np.expand_dims(output, axis=1)
        output = torch.Tensor(output)

        values, indices = torch.topk(output, 1)
        indices = indices[:, -1, 0]
        indices = indices.tolist()

        translated_sentence.append(indices)
        max_length += 1

        del output

    translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence

def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = VietOCR(len(vocab),
            config['backbone'],
            config['cnn'], 
            config['transformer'],
            config['seq_modeling'])
    
    model = model.to(device)

    return model, vocab

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def process_image(image, image_height, image_min_width, image_max_width):
    # img = image.convert('RGB')

    # w, h = img.size
    # new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    # img = img.resize((new_w, image_height), Image.Resampling.LANCZOS)

    # img = np.asarray(img).transpose(2,0, 1)
    # img = img/255
    # return img

    h, w = image.shape[:2]
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)
    img = cv2.resize(image, (new_w, image_height), interpolation=cv2.INTER_LANCZOS4)
    img = np.asarray(img).transpose(2,0,1)
    img = img/255
    return img



def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img



