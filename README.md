# Style-Transfer-with-PyTorch

Implementing the Style Transfer Paper by Gatys in PyTorch. <br/>
Link to paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
Using pretrained models to transfer arbitrary style onto arbitrary images.
The project allows one to start training and see save results from just the command line with one simple command!!

## How to use

1. Clone the repo
```
git clone https://github.com/iArunava/Style-Transfer-with-PyTorch.git
```

2. cd to the repo
```
cd Style-Transfer-with-PyTorch
```

3. Start Style Transfer
```
python3 style_transfer.py -c='/path/to/content/image' -s='/path/to/style/image'
```
4. For more involved use `--help`
```
python3 style_transfer.py -c='/path/to/content/image' -s='/path/to/style/image' -lr=0.001 -e 1000
```

**Have fun!**

## Examples

Content Image:
<img src='https://user-images.githubusercontent.com/26242097/50549153-9f071e80-0c7d-11e9-8070-ba48496aca68.jpg'/>
Photo by Olivier Guillard on Unsplash

Style Image:
<img src='https://user-images.githubusercontent.com/26242097/50549146-7d0d9c00-0c7d-11e9-8809-e6e44b3a1d8a.jpg'/>
Photo by Ashkan Forouzani on Unsplash

Stylized Image:
<img src='https://user-images.githubusercontent.com/26242097/50549118-2738f400-0c7d-11e9-8cf9-e4fdc99784b4.png'/>
Photo by an Algorithm ;)

## License

The code in this repository is distributed under GPL 3.0 <br/>
Feel free to fork the repository and perform your own style transfers!!
