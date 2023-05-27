import argparse, requests, re, json, urllib.request
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'
}

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', type=str, default='pooping dog photo',
                    help='search query')
parser.add_argument('-n', '--n_images', type=int, default=200,
                    help='number of images to collect')
parser.add_argument('-p', '--page', type=int, default=0,
                    help='page of search results to start scraping')
parser.add_argument('-d', '--dir', type=str, default='data/pooping',
                    help='directory to save images in')
parser.add_argument('-b', '--base', type=str, default='pooping',
                    help='base image name')
args = parser.parse_args()

params = {
    'q': args.query,              # search query
    'ijn': args.page,             # page number
    'tbm': 'isch',                # image results
    'hl': 'en',                   # language of the search
    'gl': 'us',                   # country where search comes from
}

def get_original_images():
    # web scrapeing function shamelessly taken from https://serpapi.com/blog/scrape-google-images-with-python/ 
    google_images = []
    all_script_tags = soup.select('script')
    matched_images_data = ''.join(re.findall(r"AF_initDataCallback\(([^<]+)\);", str(all_script_tags)))
    matched_images_data_fix = json.dumps(matched_images_data)
    matched_images_data_json = json.loads(matched_images_data_fix)

    matched_google_image_data = re.findall(r'\"b-GRID_STATE0\"(.*)sideChannel:\s?{}}', matched_images_data_json)
    matched_google_images_thumbnails = ', '.join(
        re.findall(r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]',
                   str(matched_google_image_data))).split(', ')

    thumbnails = [
        bytes(bytes(thumbnail, 'ascii').decode('unicode-escape'), 'ascii').decode('unicode-escape') for thumbnail in matched_google_images_thumbnails
    ]

    # removing previously matched thumbnails for easier full resolution image matches.
    removed_matched_google_images_thumbnails = re.sub(
        r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]', '', str(matched_google_image_data))
    matched_google_full_resolution_images = re.findall(r"(?:'|,),\[\"(https:|http.*?)\",\d+,\d+\]", removed_matched_google_images_thumbnails)

    full_res_images = [
        bytes(bytes(img, 'ascii').decode('unicode-escape'), 'ascii').decode('unicode-escape') for img in matched_google_full_resolution_images
    ]
    
    for index, (metadata, thumbnail, original) in enumerate(zip(soup.select('.isv-r.PNCib.MSM1fd.BUooTd'), thumbnails, full_res_images), start=1):
        print(f'Downloading {index} image...')        
        opener=urllib.request.build_opener()
        opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36')]
        urllib.request.install_opener(opener)

        try: 
            urllib.request.urlretrieve(original, f'{args.dir}/{args.base}_{index}.jpg')
            google_images.append({
                'title': metadata.select_one('.VFACy.kGQAp.sMi44c.lNHeqe.WGvvNb')['title'],
                'link': metadata.select_one('.VFACy.kGQAp.sMi44c.lNHeqe.WGvvNb')['href'],
                'source': metadata.select_one('.fxgdke').text,
                'thumbnail': thumbnail,
                'original': original
            })
        except Exception as e:
            print(e)

    return google_images

n_images = 0
while n_images < args.n_images:
    html = requests.get('https://www.google.com/search', params=params, headers=headers, timeout=30)
    soup = BeautifulSoup(html.text, 'html.parser')
    n_new_images = len(get_original_images())
    n_images += n_new_images
    print('PAGE {}: collected {} images'.format(params['ijn'], n_new_images))
    params['ijn'] += 1
    