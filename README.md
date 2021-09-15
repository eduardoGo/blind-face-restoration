## Project for Digital Image Process | Danilo Fernandes | Eduardo Gomes

### Paper: Towards Real-World Blind Face Restoration with Generative Facial Prior
#### Reference: [Papers with code](https://paperswithcode.com/paper/towards-real-world-blind-face-restoration) | [Paper](https://arxiv.org/abs/2103.03390)

## Dificults

The main dificult was to futher undestand the paper concepts and think about good proposals for improvements of the result, since this paper is state-of-art.

# Paper Context

Given an input facial image suffering from unknown degradation, the aim of blind face restoration is to estimate a high-quality image, which is as similar as possible to the
ground-truth image, in terms of realness and fidelity.

![image](https://user-images.githubusercontent.com/26190178/133350542-3a8aad40-c1fa-497a-b277-11f3a2a761b1.png)

Above we have a demonstration of the technique proposed by the paper used in this project. In previous works of the literature, the objective is remove the degradation of the imagens, but here, the main ideia is generate a high-quality image as similiar as possible to the real image. 

![image](https://user-images.githubusercontent.com/26190178/133350831-71e467c3-70bd-42f2-84d1-2a24d9833475.png)

The authors used a pre-trained GAN to generate the images. The GAN generates the image from latent codes (W)(which is generated by MLP that receives the vector Z from the latent space generated by the Auto-Encoder Neural Network) of the input image and from spacial features on different resolutions (realize that these spatial features are modified by SFT before concatenation and input on GAN network).

# Proposals
