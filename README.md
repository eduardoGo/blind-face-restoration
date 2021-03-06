## Project for Digital Image Process | Danilo Fernandes | Eduardo Gomes

### Paper: Towards Real-World Blind Face Restoration with Generative Facial Prior
#### Reference: [Papers with code](https://paperswithcode.com/paper/towards-real-world-blind-face-restoration) | [Paper](https://arxiv.org/pdf/2101.04061v2.pdf)

## Difficulties

The main dificult was to futher undestand the paper concepts and think about good proposals for improvements of the result, since this paper is state-of-art.

# Paper Context

Given an input facial image suffering from unknown degradation, the aim of blind face restoration is to estimate a high-quality image, which is as similar as possible to the
ground-truth image, in terms of realness and fidelity.

![image](https://user-images.githubusercontent.com/26190178/133350542-3a8aad40-c1fa-497a-b277-11f3a2a761b1.png)

Above we have a demonstration of the technique proposed by the paper used in this project. In previous works of the literature, the objective is remove the degradation of the imagens, but here, the main ideia is generate a high-quality image as similiar as possible to the real image.

![image](https://user-images.githubusercontent.com/26190178/133350831-71e467c3-70bd-42f2-84d1-2a24d9833475.png)

The authors used a pre-trained GAN to generate the images. The GAN generates the image from latent codes (W)(which is generated by MLP that receives the vector Z from the latent space generated by the Auto-Encoder Neural Network) of the input image and from spacial features on different resolutions (realize that these spatial features are modified by SFT before concatenation and input on GAN network).

# Proposal

There is some failures when the model try recovery images with scratchs.

![image](https://user-images.githubusercontent.com/26190178/133897678-be761dee-5c52-4ef5-b72c-dace21bb2a37.png)

Our proposal is apply transfer learning on original model to train with new images. This images will be degraded with textures. We hope that this way the model will overcome these limitations.

We manually download 40 textures, resize, binarize and remove background. This code can be see here https://github.com/eduardoGo/blind-face-restoration/blob/main/gfpgan/data/process_textures.py.

The code to apply degradation (degradation of the original paper + textures) to images can be see here  https://github.com/eduardoGo/blind-face-restoration/blob/main/datasets/process_validation.py.

Below we have one example of the final degradation

![image](https://user-images.githubusercontent.com/26190178/133897936-80e64843-db39-47ff-9f11-7c6cf76fc217.png)

# Results

We adapt the original codelab to run with our final model :)

Comparation with original paper, example:

![image](https://user-images.githubusercontent.com/26190178/133996738-cc3344a1-334b-4790-a82a-993a1269d9c1.png)

You can retrieve your own images with our template:

[Colab Demo](https://colab.research.google.com/drive/1TcB67AEAY3OkDKIaPfJIdkVO2WxWG51K) for GFPGAN <a href="https://colab.research.google.com/drive/1TcB67AEAY3OkDKIaPfJIdkVO2WxWG51K"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
