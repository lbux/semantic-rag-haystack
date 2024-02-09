import os
import urllib.request

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
    download_model('mistral-7b-instruct-v0.2.Q6_K.gguf', 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q6_K.gguf')