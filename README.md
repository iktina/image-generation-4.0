# Neural Image Generation version 4.0

## Welcome
The service receives a text and uses it as an input for a pre-trained model.

## What’s the point?
The service generates images that correspond to the text using machine learning methods. The service receives a text string and outputs images in binary format, representing generated images based on the received text. 

## How does it work?

This implementation involves one of two user-selectable generation models `Min-Dalle` and `Stable Diffusion`. Stable Diffusion has also been extended to the img2img implementation - `ControlNet` and `Stable Diffusion img2img`.

The user must provide a `string of text` to create images based on this string and `select a model` to generate.

The service has several parameters that allow you to select the `type of model`:

* `SD2_1`: Stable Diffusion v2.1
* `SD2`: Stable Diffusion v2.0
* `SD1_5`: Stable Diffusion v1.5
* `beksinski`: Stable Diffusion Beksinski Art
* `guohua`: Stable Diffusion Guohua
* `waifu`: Stable Diffusion Anime
* `inkpunk`: Stable Diffusion Inkpunk
* `dream_shaper`: Stable Diffusion DreamShaper
* `open_journey`: Stable Diffusion Openjourney
* `icbinp`: I Can't Believe It's Not Photography
* `dalle`: Min-Dalle
* `scribble`: ControlNet Scribble img2img
* `img2img`: Stable Diffusion img2img
* `kandinsky2_1`: Kandinsky v2.1
* `kandinsky2_2`: Kandinsky v2.2

The `model` parameter is responsible for selecting the model

Inputs:

* `metod`: image_generation
* `input_data`: 
  * a string containing the text for generation
  * a string containing `type` of a particular model
  
  For Stable Diffusion:

  * `negative_prompt` - negative prompt
  * `seedVal`- seed generation
  * `n_images` - number of images
  * `steps` - generation steps
  * `prompt_style` - add-ons for text (detailed, portrait photography realistic, animal, interior, postapocalyptic, steampunk, nature from tales, cinematic, cozy interior)


## Expected result

`Stable Diffusion v2.1`

> Prompt: cabela’s tent futuristic pop up family pod, cabin, modular, mountainous forested wilderness open fields, beautiful views, painterly concept art, joanna gaines, environmental concept art.

> Negative prompt: (deformed, distorted, disfigured:1.3), terrible view, bad anatomy, blurr, extra limb, missing limb, disgusting 

<img src=https://i5.imageban.ru/out/2023/06/23/9d1716e6bba6bc8c27d4fad562ea69a2.png width="280" > <img src=https://i3.imageban.ru/out/2023/06/23/23984396bc76fc550f3d294c7c199aeb.png width="280" >  <img src=https://i7.imageban.ru/out/2023/06/23/4c8cb87471e8e0e863bca0718ffab333.png width="280" />

`I Can't Believe It's Not Photography`

> Prompt: close up of a european woman, ginger hair, winter forest, natural skin texture, 24mm, 4k textures, soft cinematic light, RAW photo, photorealism, photorealistic, intricate, elegant, highly detailed, sharp focus, ((((cinematic look)))), soothing tones, insane details, intricate details, hyperdetailed, low contrast, soft cinematic light, dim colors, exposure blend, hdr, faded

> Negative prompt: (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation

<img src=https://i4.imageban.ru/out/2023/06/23/725c6dc11b528bd3e10c62d6dbebe889.png width="280" > <img src=https://i1.imageban.ru/out/2023/06/23/784d592d0a8380e0e79a9a1659504907.png width="280" >  <img src=https://i2.imageban.ru/out/2023/06/23/5e377ef7f4f85a4fe6178889c4f06e4b.png width="280" />

`Stable Diffusion v1.5`

> Prompt: mountains, flowers, lake, dark forest

> Addition: nature from tales

<img src=https://i4.imageban.ru/out/2023/03/13/e807343c22c79b8034905f1da19f25fe.png width="280" > <img src=https://i1.imageban.ru/out/2023/03/13/84d69ac95cdc9d518d7e4fa1b88ec697.png width="280" >  <img src=https://i5.imageban.ru/out/2023/03/13/0c09b98f70111dc6a9ec1adba159c1d2.png width="280" />

`Stable Diffusion Beksinski Art`

> Prompt: bus riding to school

<img src=https://i3.imageban.ru/out/2023/03/13/0a567f57ae1dc895edfb62513f53bc0c.png width="280" > <img src=https://i6.imageban.ru/out/2023/03/13/2b43ca452a51973f4df270abfaa7bb2e.png width="280" > <img src=https://i4.imageban.ru/out/2023/03/13/81a17f9b4a0c2c3a1bc9ddf271cfe6fc.png width="280" >

`Min-Dalle`

> Prompt: rainy sunset

<img src=https://i5.imageban.ru/out/2022/09/07/3b42e780f2787036122ab687d2bc9106.png width="280" > <img src=https://i4.imageban.ru/out/2022/09/07/fcc831f0238e3dbc2bf4c25e9214f7a0.png width="280" ><img src=https://i4.imageban.ru/out/2022/09/07/4ce30a90387f4b5ef6ddb5e7ad662c4a.png width="280" >

`Stable Diffusion img2img`

> Prompt: A fantasy landscape, trending on artstation

**Input image:**

<img src=https://i2.imageban.ru/out/2023/06/23/e48bf44563958a459812dc772a416d66.jpeg width="580" >

**Output image:**

<img src=https://i7.imageban.ru/out/2023/06/23/7af81ad46909c763b2eaa8ca2c9e050a.png width="580" >

`ControlNet Scribble img2img`

> Prompt: The building is on fire

**Input image:** / **Output image:** / **Output image:**                                                     

<img src=https://i6.imageban.ru/out/2023/06/23/571b8933bdad09ef0eaead95c45a30e0.png width="280" > <img src=https://i4.imageban.ru/out/2023/06/23/d17f3f313c426a36a46cb3f3e7952baa.png width="280" > <img src=https://i3.imageban.ru/out/2023/06/23/38387206082d7dfa88098cf96bd5a2d2.png width="280" >
