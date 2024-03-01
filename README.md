<div align="center">

# <b>ScoreHypo</b>: Probabilistic Human Mesh Estimation with Hypothesis Scoring

[Yuan Xu](https://xy02-05.github.io/)<sup>1</sup>, [Xiaoxuan Ma](https://shirleymaxx.github.io/)<sup>1</sup>, [Jiajun Su](https://scholar.google.com/citations?user=DoUvUz4AAAAJ&hl=zh-CN)<sup>2</sup>, [Wentao Zhu](https://wentao.live/)<sup>1</sup>, [Yu Qiao](http://www.pami.sjtu.edu.cn/yuqiao)<sup>3</sup>, [Wentao Zhu](https://wentao.live/)<sup>1</sup>

<sup>1</sup>Peking University <sup>2</sup>International Digital Economy Academy (IDEA) <sup>3</sup>Shanghai Jiao Tong University

### [Projectpage](https://xy02-05.github.io/ScoreHypo/) · [Paper (Coming Soon)] · [Video (Coming Soon)]

</div>


***Abstract**: Monocular 3D human mesh estimation is an ill-posed problem, characterized by inherent ambiguity and occlusion. While recent probabilistic methods propose generating multiple solutions, little attention is paid to obtaining high-quality estimates from them. To address this limitation, we introduce <b>ScoreHypo</b>, a versatile framework by first leveraging our novel <b>HypoNet</b> to generate multiple hypotheses, followed by employing a meticulously designed scorer, <b>ScoreNet</b>, to evaluate and select high-quality estimates. ScoreHypo formulates the estimation process as a reverse denoising process, where HypoNet produces a diverse set of plausible estimates that effectively align with the image cues. Subsequently, ScoreNet is employed to rigorously evaluate and rank these estimates based on their quality and finally identify superior ones. Experimental results demonstrate that HypoNet outperforms existing state-of-the-art probabilistic methods as a multi-hypothesis mesh estimator. Moreover, the estimates selected by ScoreNet significantly outperform random generation or simple averaging. Notably, the trained ScoreNet exhibits generalizability, as it can effectively score existing methods and significantly reduce their errors by more than 15%.*

Code is coming soon.





