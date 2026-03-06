from torch.nn.modules import loss

import DSH
from model.hash_model import DCMHT as DCMHT
# from model.hash_model import Hier_Model
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as scio
from .base import TrainBase
from model.optimization import BertAdam
from utils import get_args, calc_neighbor, cosine_similarity, euclidean_similarity
from utils.calc_utils import calc_map_k_matrix as calc_map_k
from dataset.dataloader import dataloader
import numpy as np
from QuadrupletMarginLoss import QuadrupletMarginLoss
from DSH import DSHLoss
from alex import AlexNet

from torch.nn.functional import normalize
class Trainer(TrainBase):

    def __init__(self,
                 rank=1):
        args = get_args()
        super(Trainer, self).__init__(args, rank)
        # bit = args.output_dim
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        linear = False
        if self.args.hash_layer == "linear":
            linear = True

        self.logger.info("ViT+GPT!")
        HashModel = DCMHT
        self.ClassLen = 0
        if self.args.dataset == 'nuswide':
            self.ClassLen = 21
        elif self.args.dataset == 'flickr25k':
            self.ClassLen = 24
        elif self.args.dataset == 'coco':
            self.ClassLen = 80
        else:
            self.ClassLen = 291
        # self.MSL = MultiSimilarityLoss().to(0)
        self.Quad = QuadrupletMarginLoss().to(1)
        self.qua = nn.MSELoss().to(1)
        self.CE= nn.CrossEntropyLoss().to(1)
        self.dsh = DSH.DSHLoss(self.args.output_dim).to(1)
        self.dtsh = DSH.DTSHLoss().to(1)
        self.alex = AlexNet(hash_bit=self.args.output_dim).to(1)
        # from loss import MarginLoss
        # from config import get_config
        # args = get_config()
        # self.Loss = MarginLoss(args)
        # self.L_net = LabelNet(code_len=self.args.output_dim , label_dim=self.ClassLen).to(self.rank)
        # self.L_opt = torch.optim.SGD(self.L_net.parameters(),lr=1e-2,momentum=0.9,weight_decay=1e-5)
        self.lossMS_I2T_list = np.array([])
        self.lossMS_I2I_list = np.array([])
        self.lossMS_T2T_list = np.array([])


        self.model = HashModel(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                               writer=self.writer, logger=self.logger, is_train=self.args.is_train, linear=linear).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))



        self.model.float()
        # self.supervision.float()


        self.optimizer = BertAdam([
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
            {'params': self.Quad.parameters(), 'lr': self.args.lr},
            # {'params': self.qua.parameters(), 'lr': self.args.lr},
            {'params': self.CE.parameters(), 'lr': self.args.lr},
            {'params': self.dsh.parameters(), 'lr': self.args.lr},
            {'params': self.dtsh.parameters(), 'lr': self.args.lr},
            {'params': self.alex.parameters(), 'lr': self.args.lr},

            # {'params': self.MSL.parameters(), 'lr': self.args.lr},
            # {'params': self.L_net.parameters(),'lr':self.args.lr * 10},
        ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        # print(self.model)

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file,
                                                            indexFile=self.args.index_file,
                                                            labelFile=self.args.label_file,
                                                            maxWords=self.args.max_words,
                                                            imageResolution=self.args.resolution,
                                                            query_num=self.args.query_num,
                                                            train_num=self.args.train_num,
                                                            seed=self.args.seed)
        self.train_dataset = train_data
        self.train_labels = train_data.get_all_label()
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def train_epoch(self, epoch):

        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        all_loss = 0
        i_lossl = 0
        t_lossl = 0
        it_lossl = 0
        times = 0
        scaler = torch.cuda.amp.GradScaler()
        
        for image, text, label, index in self.train_loader:
            self.global_step += 1
            times += 1
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            labels = label.float().to(self.rank)
            
            # self.L_net.set_alpha(epoch)
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                hash_img, I, hash_text, T = self.model(image, text)
                # _, I, hash_text, T = self.model(image, text)

                # hash_img = self.alex(image)
                # print(len(labels))
                i,n = self.Quad(hash_img, labels, hash_img)
                t,m = self.Quad(hash_text, labels, hash_text)
                it,nm = self.Quad(hash_img, labels, hash_text)
                # i = self.dsh(hash_img,labels,hash_img)
                # t = self.dsh(hash_text, labels, hash_text)
                # it = self.dsh(hash_img, labels, hash_text)
                #i = self.dtsh(hash_img,labels,hash_img)
                #t = self.dtsh(hash_text, labels, hash_text)
                #it = self.dtsh(hash_img, labels, hash_text)
                ii = self.CE(I,labels)
                tt = self.CE(T,labels)
                balance_loss = 0.1 * ((labels.mean(dim=0) - 0.5).abs().mean())
                # iii = self.qua(torch.abs(hash_img), torch.ones(hash_img.data.shape).cuda(1))
                # ttt = self.qua(torch.abs(hash_text), torch.ones(hash_text.data.shape).cuda(1))

                tot_hier = i+t+it+0.1*(ii+tt)+0.3*balance_loss
                # tot_hier = i + t + it + 0.3 * balance_loss
            scaler.scale(tot_hier).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # all_loss += img_loss1 + i_t_loss1 + text_loss1 + loss1 + loss2 + label_loss1
            all_loss += tot_hier
            # i_lossl += img_loss1.cpu().item()
            # t_lossl += text_loss1.cpu().item()
            # it_lossl += i_t_loss1.cpu().item()
            

            
            # tot_hier.backward()

            # self.optimizer.step()
        self.lossMS_I2I_list = np.append(self.lossMS_I2I_list, i_lossl / len(self.train_loader))
        self.lossMS_T2T_list = np.append(self.lossMS_T2T_list, t_lossl / len(self.train_loader))
        self.lossMS_I2T_list = np.append(self.lossMS_I2T_list, it_lossl / len(self.train_loader))
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

    def train(self):
        self.logger.info("Start train.")


        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.valid(epoch)
            self.save_model(epoch)
        np.savetxt("TrainMS_M_III", self.lossMS_I2I_list, delimiter=',')
        np.savetxt("TrainMS_M_TTT", self.lossMS_T2T_list, delimiter=',')
        np.savetxt("TrainMS_M_IT", self.lossMS_I2T_list, delimiter=',')
        self.logger.info(
            f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")

    def test(self, model='i2t'):
        self.logger.info("test")
        self.change_state(mode="valid")
        query_img, query_txt = super().get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt = super().get_code(self.retrieval_loader, self.args.retrieval_num,
                                                        )
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)


    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt = super().get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt =  super().get_code(self.retrieval_loader, self.args.retrieval_num,)
        # print("get all code")
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t", map=mAPi2t)
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i", map=mAPt2i)
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t", map=0):
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)
        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir,
                                  str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + f'{map:.4f}_.mat'),
                     result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")


