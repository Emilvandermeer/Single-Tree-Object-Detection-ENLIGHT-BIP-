def main():
    print("Hello from bip-proj!")
    load_files()

def load_files():

    import os
    import requests
    import zipfile

    def get_owncloud_file(url, folder, target_dir="data"):
        
        # Create target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        file_path = os.path.join(target_dir, folder)

        # Check if file exists already so we dont have to downlaod again
        if os.path.exists(file_path):
            return file_path

        print(f"Downloading '{folder}' from ownCloud...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for HTTP errors
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("Download complete.")
            return file_path
        except Exception as e:
            print(f"Failed to download file: {e}")
            return None

    # Usage example
    OWNCLOUD_URL = "https://owncloud.gwdg.de/index.php/s/H6MsR0wVGRuPPl3/download"
    FILENAME = "dataset"

    get_owncloud_file(OWNCLOUD_URL,FILENAME)

    def unzip1():
        extract_to = "data_unzip"
        zip_path = "data/dataset"

        if os.path.exists(extract_to) and os.listdir(extract_to):
            print(f"Extraction skipped. '{extract_to}' already exists.")
            return

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    unzip1()

    def unzip2():
        extract_to2 = "data_unzip2"
        zip_path2= "data_unzip/neon_tree/NeonTreeEvaluation.zip"
        

        if os.path.exists(extract_to2) and os.listdir(extract_to2):
            print(f"Extraction skipped. '{extract_to2}' already exists.")
            return

        print(f"Extracting {zip_path2}...")
        with zipfile.ZipFile(zip_path2, 'r') as zip_ref:
            zip_ref.extractall(extract_to2)
    unzip2()

    def unzip3():
        extract_to1 = "annotations"
        zip_path1= "data_unzip2/annotations.zip"

        #ANNOTATIONS UNZIP

        if os.path.exists(extract_to1) and os.listdir(extract_to1):
            print(f"Extraction skipped. '{extract_to1}' already exists.")
            return

        print(f"Extracting {zip_path1}...")
        with zipfile.ZipFile(zip_path1, 'r') as zip_ref:
            zip_ref.extractall(extract_to1)

        #EVALUATION UNZIP
        extract_to2 = "evaluation"
        zip_path2= "data_unzip2/evaluation.zip"
        
        if os.path.exists(extract_to2) and os.listdir(extract_to2):
            print(f"Extraction skipped. '{extract_to2}' already exists.")
            return

        print(f"Extracting {zip_path2}...")
        with zipfile.ZipFile(zip_path2, 'r') as zip_ref:
            zip_ref.extractall(extract_to2)
        

        #TRAINING UNZIP
        extract_to3 = "training"
        zip_path3= "data_unzip2/training.zip"

        if os.path.exists(extract_to3) and os.listdir(extract_to3):
            print(f"Extraction skipped. '{extract_to3}' already exists.")
            return

        print(f"Extracting {zip_path3}...")
        with zipfile.ZipFile(zip_path3, 'r') as zip_ref:
            zip_ref.extractall(extract_to3)
    unzip3()

    



    

    
    # Optional: remove the zip file after successful extraction to save space
    # os.remove(zip_path)

    

if __name__ == "__main__":
    main()
