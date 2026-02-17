# [Deep Potential Semantic-aware Hashing for for Cross-modal Retrieval](https://www.sciencedirect.com/science/article/abs/pii/S0952197626004367)
This paper is accepted for publication with EAAI.


## Training

### Processing dataset
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH)

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start

After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128

### Citation
@ARTICLE{wu2026deep,  
  author={Wu, Lei and Qin, Qibing and Dai, Jiangyan and Huang, Lei and Zhang, Wenfeng},  
  journal={Engineering Applications of Artificial Intelligence},  
  title={Deep Potential Semantic-aware Hashing for Cross-modal Retrieval},  
  year={2026},  
  volume={169},  
  number={11},  
  pages={114115},  
  doi={https://doi.org/10.1016/j.engappai.2026.114155}}
