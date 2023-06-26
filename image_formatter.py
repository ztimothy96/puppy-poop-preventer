import argparse
import os
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='./',
                    help='directory to get images from')
parser.add_argument('-o', '--output', type=str, default='./',
                    help='directory to save images in')
parser.add_argument('-b', '--base', type=str, default='image',
                    help='base image name')
parser.add_argument('-f', '--format', type=str, default='png',
                    help='format to save image in')
args = parser.parse_args()

exts = ['.png', '.jpg', '.jpeg', '.webp']

def process_image(path, count):
    original = Image.open(path)
    original.save('{}{}_{}.{}'.format(args.output, args.base, count, args.format), format=args.format)
    return count + 1

def main():
    count = 1
    im_names = sorted([file for file in os.listdir(args.input)
                       if any([file.endswith(ext) for ext in exts])])
    for file_name in im_names:
        file_path = args.input + file_name
        count = process_image(file_path, count) 
        
    return True

if __name__ == "__main__":
    main()