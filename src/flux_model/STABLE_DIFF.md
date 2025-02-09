## Stable Diffusion for Image generation 🌄
As already mentioned [here](https://github.com/kulendu/stable-diffusion-pipeline/blob/master/README.md), for generating images, **Stable Diffusion** is used from *🧨[diffusers](https://github.com/huggingface/diffusers)*

Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/) and [LAION](https://laion.ai/). It's trained on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and can run on many consumer GPUs.
See the [model card](https://huggingface.co/CompVis/stable-diffusion) for more information.

This [Colab notebook](https://colab.research.google.com/drive/1unuR9Ta4i7qlV-Ll9RfBjkyN9dEA0Xwf?usp=sharing) shows how to use Stable Diffusion with the 🤗 Hugging Face [🧨 Diffusers library](https://github.com/huggingface/diffusers). 

## Stable Diffusion Pipeline
`StableDiffusionPipeline` is an end-to-end inference pipeline that can be used to generate images from text with just a few lines of code. 

The notebook demonstrated uses Stable Diffusion version 1.4 [CompVis/stable-diffusion-v1-4](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2FCompVis%2Fstable-diffusion-v1-4) as the pretrained weights.

```cpp
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

```

## Running the Diffusion pipeline
`mannual_seed = 1024` is used for generating same set of images, eveytime executing the cell. The prompt used for generation is: `"a man riding bike in a futuristic city with a neon lights in the background"`

```cpp
num_images = 5
prompt = ["a man riding bike in a futuristic city with a neon lights in the background"] * num_images
generator = torch.Generator(device).manual_seed(1024)

images = pipeline(prompt, num_inference_steps=30, generator=generator).images
```

This generated the following 5 images

![](../../images/all_generated_images.png)