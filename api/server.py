# Flask Imports
from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import jsonify, request
from flask import flash

from pipeline import Inference
from utils import *
import cv2

app = Flask(__name__)
inference = Inference('maskrcnn_resnet_50.pt')

@app.route('/', methods=['GET'])
def index():
    print('helloworld')
    return '1'

@app.route('/upload', methods = ['POST'])
def results():
    #get the image
    image_file = request.files['image']

    #save the image
    rand_img_name = str(randomString())
    image_out_file = 'input_images/' + rand_img_name + '.png'
    image_file.save(image_out_file)

    #get masks
    masked_img = inference.get_pred(image_out_file)

    #save the output img
    cv2.imwrite('output_images/' + rand_img_name + '.png',
                cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

    return 'success'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = True)
