## Image Diffusion
`Task:` Take two images, and using Diffusion Models to merge them while keeping the semantics same to generate a single image. Image (*left*) of a person whose T-shirt needed to be changed, and image (*right*) of the T-shirt to which it should change.

![alt text](image_merged_manually.png)

### Data Tweaking
Using image of the t-shirt (*right*), as the guided image, the model fail to produce accurate drape on the model's image. So to overcome this, same image of the Tshirt from [overlays](https://overlaysnow.com/products/be-the-change-navy-blue-relaxed-fit-t-shirt-ultra-soft-copy) have been taken and the further segmented from the body to create a more detailed and zoomed guidance for the model to preseve the high frequence details. 

<img src="./image-removebg-preview.png" style="height:20em;"></img>

### Experiments
1. **Flux Dev/Schell**: Family of Flux models such as: [Flux.1 [schell]](https://huggingface.co/black-forest-labs/FLUX.1-schnell) and [FLUX.1 [dev]](https://huggingface.co/black-forest-labs/FLUX.1-dev) are great, and also can produce high quality images, but the bottleneck comes in computation. So [Flux.1 [schell]](https://huggingface.co/black-forest-labs/FLUX.1-schnell) is definately faster compared to [FLUX.1 [dev]](https://huggingface.co/black-forest-labs/FLUX.1-dev), but schell can run on macOS (Metal chips), but again the runtime is very slow, and also crashes sometimes, depending on the quality of prompt given.

2. **Stable Diffusion/ControlNet**:

3. **OOTDiffusion**: 

