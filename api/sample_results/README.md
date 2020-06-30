### Description

This repository provides an API to run inference on the trained MaskRCNN model. Find the details of the contents below-
* `server.py` : This file defines the API, which is built on `Flask`.
* `pipeline.py` : This is the main python file and contains the Inference class.
* `models.py` : This file defines the modified model for our use case.
* `utils.py` : This file contains a mix of helper functions for `pipeline.py`.
* `sample_results/` : This folder contains some sample results of the API.
* `requirements.txt` : This file lists the dependencies needed to run the code.
* `input_images/` : This folder contains images, to be tested.
* `output_images/` : The output image of the API is written in this folder.

Note: This repository use `Python 3.6`.<br/>
Note: This repository was built on a MAC machine so the inference function does not support CUDA and is performed on CPU.

### Steps to run the API

1. Clone the repository and go inside the directory.
2. Run `pip install -r requirements.txt` to install the dependencies. 
3. Donwload the model in the root directory from [here](https://drive.google.com/file/d/1yHgibFxDNkXlHO3aI4OtLdyTSJgjYI3S/view?usp=sharing) as `maskrcnn_resnet_50.pt`.
3. Run `python server.py` on terminal in the current directory. This will open our port to receive requests.
4. Run the following `curl` command to send API request. 

                  curl --location --request POST 'localhost:5000/upload' --form 'image=@/path/to/file'
                  
    This would save the image file to `input_images/` and write the output to `output_images/`. The output image will dispay the masks and bounding boxes as well as the precentage of masked pixels on the upper left corner.
    
  
