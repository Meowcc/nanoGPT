import os

# train data: Harry Potter
def prepare_data(data_dir):
    data = ""
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):  
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data += f.read()

    return data