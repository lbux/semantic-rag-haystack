import os
import urllib.request

from constants import MODEL_NAME, MODEL_GGUF_URL

def download_model(model_name, model_url):
    model_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        print('Downloading model...')
        urllib.request.urlretrieve(model_url, model_path)
        print('Model downloaded.')
    else:
        print('Model already exists.')

# call the function with parameters and allow to be ran from script
if __name__ == '__main__':
    download_model(MODEL_NAME, MODEL_GGUF_URL)