from google_images_download import google_images_download

def download_images(classes, output_dir, limit=1000):
    response = google_images_download.googleimagesdownload()
    for class_name in classes:
        arguments = {
            "keywords": class_name,
            "limit": limit,
            "print_urls": False,
            "output_directory": output_dir,
            "image_directory": class_name,
            "format": "jpg",
            "safe_search": True
        }
        try:
            response.download(arguments)
            print(f"Downloaded images for class: {class_name}")
        except Exception as e:
            print(f"Error downloading images for class {class_name}: {e}")

if __name__ == "__main__":
    classes = [
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    ]  # CIFAR-10 classes
    output_dir = "internet_reference"
    download_images(classes, output_dir)