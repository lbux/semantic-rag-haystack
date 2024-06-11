import os


def create_source_path():
    folder_path = "documents/source_documents"
    source_documents_folder = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    return source_documents_folder
