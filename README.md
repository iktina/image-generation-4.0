# Neural Image Generation version 2.0

## Welcome
The service receives a text and uses it as an input for a pre-trained model.

## What’s the point?
The service generates images that correspond to the text using machine learning methods. The service receives a text string and outputs images in binary format, representing generated images based on the received text. 

## How does it work?

This implementation assumes the use of one of two generation models at the user's choice `Min-Dalle` and `Stable Diffusion`.

The user must provide a `string of text` to create images based on this string and `select a model` to generate.

The service has several parameters that allow you to select the `type of model`:

* `dalle`: Min-Dalle
* `Stable Diffusion`: Stable Diffusion
* `beksinski`: Stable Diffusion Beksinski Art
* `guohua`: Stable Diffusion Guohua
* `waifu`: Stable Diffusion Anime

If parameter is active (1), then the model will be used.

Inputs:

* `metod`: image_generation
* `input_data`: 
  * a string containing the text for generation
  * binary variable `type` to activate a particular model
  
  For Stable Diffusion:
  
  * `seed`- seed generation
  * `n_images` - number of images
  * `steps` - generation steps
  * `H` - image height
  * `W` - image width
  * `prompt_style` - add-ons for text (detailed, portrait photography realistic, animal, interior, postapocalyptic, steampunk, nature from tales, cinematic, cozy interior)
  
## Expected result

`Min-Dalle`

> Text: rainy sunset

<img src=https://i5.imageban.ru/out/2022/09/07/3b42e780f2787036122ab687d2bc9106.png width="280" > <img src=https://i4.imageban.ru/out/2022/09/07/fcc831f0238e3dbc2bf4c25e9214f7a0.png width="280" ><img src=https://i4.imageban.ru/out/2022/09/07/4ce30a90387f4b5ef6ddb5e7ad662c4a.png width="280" >

`Stable Diffusion`

> Text: mountains, flowers, lake, dark forest

> Addition: nature from tales

<img src=https://i4.imageban.ru/out/2023/03/13/e807343c22c79b8034905f1da19f25fe.png width="280" > <img src=https://i1.imageban.ru/out/2023/03/13/84d69ac95cdc9d518d7e4fa1b88ec697.png width="280" >  <img src=https://i5.imageban.ru/out/2023/03/13/0c09b98f70111dc6a9ec1adba159c1d2.png width="280" />

`Stable Diffusion Beksinski Art`

> Text: bus riding to school

<img src=https://i3.imageban.ru/out/2023/03/13/0a567f57ae1dc895edfb62513f53bc0c.png width="280" > <img src=https://i6.imageban.ru/out/2023/03/13/2b43ca452a51973f4df270abfaa7bb2e.png width="280" > <img src=https://i4.imageban.ru/out/2023/03/13/81a17f9b4a0c2c3a1bc9ddf271cfe6fc.png width="280" >
