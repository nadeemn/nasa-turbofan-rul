import kagglehub

class DatasetDownloader:
    def __init__(self, dataset_name:str):
        self.dataset_name = dataset_name
        self.download_path = None

    def download(self):
        self.download_path = kagglehub.dataset_download(self.dataset_name)
        return self.download_path
    
if __name__ == "__main__":
    dataset_name = "behrad3d/nasa-cmaps"
    downloader = DatasetDownloader(dataset_name)
    path = downloader.download()
    print("path to dataset", path)