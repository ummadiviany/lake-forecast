import os
from glob import glob

for dir in glob('data/*'):
    print('-'*50)
    files = sorted(glob(f'{dir}/train/*.png')) + sorted(glob(f'{dir}/test/*.png'))
    get_name = lambda x: x.split('\20')[-1]
    files = sorted(map(get_name, files))
    print(f'Loaded {len(files)} images from {dir} dataset')
    print(f'First image: {files[0]}')
    print(f'Last image: {files[-1]}')