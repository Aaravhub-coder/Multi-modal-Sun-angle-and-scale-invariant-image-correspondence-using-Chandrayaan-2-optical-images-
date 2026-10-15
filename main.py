### TOP LEVEL CODE ###

from interface import get_input_paths
from preprocess import preprocess_pair
from baseline import run_baseline

def main():
    source_image_path, reference_image_path = get_input_paths()
    if not source_image_path or not reference_image_path:
        print("Both a source image and a reference file are required. Exiting.")
        exit(1)

    print("Preprocessing images")
    src, ref = preprocess_pair(source_image_path, reference_image_path)

    ### Next: baseline matching goes here (src, ref are preprocessed Img objects)
    result = run_baseline(src,ref)
    print(result)

if __name__ == "__main__":
    main()