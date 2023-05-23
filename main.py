from flask import Flask,request,render_template,redirect
import os

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch

from model import Net
from utils import ConfigS, ConfigL, download_weights

config = ConfigL()

# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'

img_save_path = ""

app = Flask(__name__)


app.config["IMAGE_UPLOADS"] = "/home/sacchin/Desktop/html/static/Images"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]

from werkzeug.utils import secure_filename


@app.route('/home',methods = ["GET","POST"])
def upload_image():
	if request.method == "POST":
		image = request.files['file']
		if image.filename == '':
			print("Image must have a file name")
			return redirect(request.url)
		filename = secure_filename(image.filename)
		basedir = os.path.abspath(os.path.dirname(__file__))
		image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))
		print("such "+os.path.join(basedir,app.config["IMAGE_UPLOADS"]+filename))
		img_path = os.path.join(basedir,app.config["IMAGE_UPLOADS"]+"/"+filename)

		ckp_path = os.path.join(config.weights_dir, "sacchin_large")
		print(img_path)
		assert os.path.isfile(img_path), 'Image does not exist'
		
		img = Image.open(img_path)

		model = Net(
			clip_model=config.clip_model,
			text_model=config.text_model,
			ep_len=config.ep_len,
			num_layers=config.num_layers, 
			n_heads=config.n_heads, 
			forward_expansion=config.forward_expansion, 
			dropout=config.dropout, 
			max_len=config.max_len,
			device=device
		)

		if not os.path.exists(config.weights_dir):
			os.makedirs(config.weights_dir)

		if not os.path.isfile(ckp_path):
			download_weights(ckp_path, 'S')
			
		checkpoint = torch.load(ckp_path, map_location=device)
		model.load_state_dict(checkpoint)

		model.eval()

		with torch.no_grad():
			caption, _ = model(img, 1.0)
		
		plt.imshow(img)
		plt.title(caption)
		plt.axis('off')
		global img_save_path
		img_save_path = os.path.join(basedir,app.config["IMAGE_UPLOADS"]+"/L"+filename)
		plt.savefig( img_save_path, bbox_inches='tight')

		plt.clf()
		plt.close()

		print('Generated Caption: "{}"'.format(caption))
		print(filename)

		return render_template("main.html",filename="L"+filename)
	return render_template('main.html')


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static',filename = img_save_path), code=301)


app.run(debug=True,port=2000)
