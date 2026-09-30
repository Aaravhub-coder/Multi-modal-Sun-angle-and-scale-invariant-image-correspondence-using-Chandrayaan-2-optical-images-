from interface import get_input_paths

def preprocess(source_image_path, reference_image_path):
    print(f"Preprocessing Images")
    print(f"Source Image: {source_image_path}")
    print(f"Reference Image: {reference_image_path}")
    ### Write preprocessing code here

def main():
    source_image_path , reference_image_path = get_input_paths()
    if not source_image_path or not reference_image_path:
        print("Both a source image and a reference file are required. Exiting.")
        exit(1)
    preprocess(source_image_path, reference_image_path)
    ### All functions to be ultimately called here after preprocessing is done

if __name__ == "__main__":
    main()





