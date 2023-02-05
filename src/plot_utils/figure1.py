import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

paths = {
    'mead' : {'mead0' : 'data/mead/test/20220505T181921.png',
            'mead1' : 'data/mead/train/20190605T181929.png',},
    'puzhal' : {'puzhal0' : 'data/puzhal/train/20190411T045701.png',
            'puzhal1' : 'data/puzhal/train/20220311T045659.png',},
    'sawa' : {'sawa0' : 'data/sawa/train/20211219T074229.png',
            'sawa1' : 'data/sawa/train/20191031T074009.png',},
    'bellandur' : {'bellandur0' : 'data/bellandur/train/20220418T050701.png',
            'bellandur1' : 'data/bellandur/train/20181220T051219.png'}
    
}

c = 3
fig = plt.figure(figsize=(4*c, 2*c))

for i, (name, subdict) in enumerate(paths.items()):
    v0 = Image.open(subdict[f'{name}0'])
    v1 = Image.open(subdict[f'{name}1'])
    get_data = lambda x : x.split("/")[-1][:4] + '-' + x.split("/")[-1][4:6]
    
    if name == 'bellandur':
        v0 = v0.rotate(90, expand=True)
        v1 = v1.rotate(90, expand=True)
    
    plt.subplot(2, 4, i+5)
    plt.imshow(v0)
    plt.title(f'{get_data(subdict[f"{name}0"])}')
    plt.axis('off')
    plt.subplot(2, 4, i+1)
    plt.imshow(v1)
    plt.title(f'{get_data(subdict[f"{name}1"])}')
    plt.axis('off')
    
# plt.tight_layout()
# plt.suptitle('Lake images', fontsize=16, y=0.85)
# plt.show()
# st = fig.suptitle("Lake Dataset Samples")
plt.savefig("artifacts/plots/figure1.png",  bbox_inches='tight', dpi=200)

