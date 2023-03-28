import sys
import io
from PIL import Image
import cv2
import base64
import torch
from flask import Flask, render_template, request, make_response
from werkzeug.exceptions import BadRequest
import os
import json

# creating flask app
app = Flask(__name__)


# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {}
# create a list of keys to use them in the select part of the html code
listOfKeys = []

# inference fonction
def get_prediction(img_bytes,model):
    img = Image.open(io.BytesIO(img_bytes))
    # inference
    results = model(img, size=640)  
    # extract the predicted class labels, confidence scores, and bounding box coordinates
    # class_labels = results.pred[0][:, -1].numpy()
    # conf_scores = results.pred[0][:, -2].numpy()
    # bbox_coords = results.pred[0][:, :-2].numpy()
    return results


# get method
@app.route('/', methods=['GET'])
def get():
    # in the select we will have each key of the list in option
    return render_template("index.html", len = len(listOfKeys), listOfKeys = listOfKeys)

@app.route('/pred', methods=['POST'])
def pred():
    print("Hello World")
    return render_template("hello.html")

# post method
@app.route('/', methods=['POST'])
def predict():
    file = extract_img(request)
    img_bytes = file.read()
    
    # choice of the model
    model_get = dictOfModels[request.form.get("model_choice")]
    results = get_prediction(img_bytes,model_get)
    print(f'User selected model : {request.form.get("model_choice")}')

#     class_labels = results.pred[0][:, -1].numpy()
#     conf_scores = results.pred[0][:, -2].numpy()
#     bbox_coords = results.pred[0][:, :-2].numpy()

#     output_dict = {
#     'objects': [
#         {
#             'class': model_get.names[int(class_labels)]
#             'confidence': float(conf_scores),
#             'bbox': bbox.tolist()
#         } for cls, conf, bbox in zip(results.pred[0][:, -1], results.pred[0][:, 4], results.pred[0][:, :4])
#     ]
#   }
    # extract predicted class and coordinates
    labels = results.xyxy[0][:, -1].tolist()
    boxes = results.xyxy[0][:, :-1].tolist()
    conf_scores = results.pred[0][:, -2].tolist()
    # create string with class and coordinates
    output_str = ''
    results_json = []
    for i in range(len(labels)):
        out_name = results.names[int(labels[i])].capitalize()
        out_conf = conf_scores[i]
        output_str += f'{out_name}: conf: {round(out_conf,3)}, at {round(boxes[i][0])}, {round(boxes[i][1])}, {round(boxes[i][2])}, {round(boxes[i][3])}\n'
        results_json.append({
            "class": out_name,
            "confidence": float(out_conf),
            "topleft": boxes[i][0],
            "bottomright": boxes[i][3]
        })

    output_str_break = output_str.replace("\n", "<br>")
    json_out  = json.dumps(results_json, indent=4)

    # updates results.imgs with boxes and labels
    results.render()
    
    # encoding the resulting image and return it
    for img in results.ims:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_arr = cv2.imencode('.jpg', RGB_img)[1]
        # response = make_response(im_arr.tobytes())
        encoded_image_base64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')
        # response.headers['Content-Type'] = 'image/jpeg'
    return render_template("index.html", len = len(listOfKeys), listOfKeys = listOfKeys, responsed = encoded_image_base64, output_str=output_str_break, json_out=json_out);
    # return response;

def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")
        
    file = request.files['file']
    
    if file.filename == '':
        raise BadRequest("Given file is invalid")
        
    return file
    

if __name__ == '__main__':
    print('Starting yolov5 webservice...')
    # Getting directory containing models from command args (or default 'models_train')
    models_directory = 'models_train'
    if len(sys.argv) > 1:
        models_directory = sys.argv[1]
    print(f'Watching for yolov5 models under {models_directory}...')
    for r, d, f in os.walk(models_directory):
        for file in f:
            if ".pt" in file:
                # example: file = "model1.pt"
                # the path of each model: os.path.join(r, file)
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(r, file)
                print(f'Loading model {model_path} with path {model_path}...')
                dictOfModels[model_name] = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
                # you would obtain: dictOfModels = {"model1" : model1 , etc}
        for key in dictOfModels :
            listOfKeys.append(key) # put all the keys in the listOfKeys

    #print(f'Server now running on {os.environ["JOB_URL_SCHEME"]}{os.environ["JOB_ID"]}.{os.environ["JOB_HOST"]}')
    print('Server is now running')
    
    # starting app
    app.run(debug=True,host='0.0.0.0')