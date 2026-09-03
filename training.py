import os
import shutil
import torch
from tqdm import tqdm
import math
from validation import score_anomalies

# import wandb


class SeeKerTrainer:
    def __init__(self, args, model, train_loader, test_loader, val_loader, val_metadata, test_metadata,
                 optimizer_f=None, log_writer=None, dataset="UBnormal"):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_metadata = val_metadata
        self.test_metadata = test_metadata
        self.test_loader = test_loader
        self.log_writer = log_writer
        self.optimizer = optimizer_f(self.model.parameters())
        self.dataset = dataset

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = f'checkpoint_{epoch}.pth'

        state['args'] = self.args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = filename
        try:
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # self.model.set_actnorm_init()
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(filename, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    

    def joint_lnp(self, x):
        B, T, N, A = x.shape
        out = self.model(x)

        mu, logvar = torch.chunk(out, 2, dim=1)

        mu = mu.reshape(B, T * N, A)[:, (-N-1):-1]
        logvar = logvar.reshape(B, T * N, A)[:, (-N-1):-1]

        # Clamp log-variance to avoid numerical overflow/underflow
        # logvar = torch.clamp(logvar, min=-6, max=2) # < new

        # var = torch.exp(logvar)
        E_inv = torch.diag_embed(torch.exp(-logvar))

        x = x.reshape(B, T * N, A)
        x_tgt = x[:, -N:]

        lnp = -0.5 * (
            torch.einsum('bti,btij,btj->bt', x_tgt - mu, E_inv, x_tgt - mu)
            + torch.sum(logvar, dim=-1) + x_tgt.shape[-1] * math.log(math.pi * 2))
        
        # lnp = torch.einsum('bti,btij,btj->bt', x_tgt - mu, E_inv, x_tgt - mu) + torch.sum(logvar, dim=-1) + x_tgt.shape[-1] * math.log(math.pi * 2)

        return lnp
    


    def train(self, clip=100):
        start_epoch = 0
        num_epochs = self.args.epochs
        self.model.train()
        self.model = self.model.to(self.args.device)
        all_val_auc = {}

        # print(vars(self.args))
        # exit()

        # start wandb run 
        run = wandb.init(
            entity="hoanghung17jan-vu-hoang-hung",
            project="Project",
            config={
                "name": "latest",
                **vars(self.args),
            }
        )

        for epoch in range(start_epoch, num_epochs):
            self.model.train()

            print("Starting Epoch {} / {}".format(epoch + 1, num_epochs))
            pbar = tqdm(self.train_loader)
            for _, data_arr in enumerate(pbar):
                x = torch.permute(data_arr[0], (0,2,3,1))[..., :2]
                x = x.to(self.args.device, non_blocking=True) 
                
                lnp = self.joint_lnp(x.float())

                negloglik_loss = - lnp.sum(-1).mean()
                negloglik_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

                pbar.set_description("Loss: {}".format(negloglik_loss.item()))

            self.log_writer.add_scalar('NLL Loss', negloglik_loss.item(), epoch )

            if self.dataset not in ["ShanghaiTech", "MSAD"]:
                auc_val = self.validate()
                all_val_auc[epoch] = auc_val
            else:
                auc_val = 0

            run.log({"auc": auc_val, "loss": negloglik_loss.item(), })
            
            is_best = auc_val == max(all_val_auc.values(), default=0)
            if is_best:
                self.save_checkpoint(epoch=epoch, filename="checkpoint_best.pth")


        run.finish()

        return auc_val
    
    def validate(self):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.val_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Eval")
        for _, data_arr in enumerate(pbar):
            # print(data_arr)
            # exit()

            x = torch.permute(data_arr[0], (0,2,3,1))[..., :2]
            x = x.to(self.args.device, non_blocking=True) 

            conf = torch.permute(data_arr[0], (0,2,3,1))[..., -1]
            B, T, N, _ = x.shape

            # print(x.shape)
            # exit()

            with torch.no_grad():
                lnp =  self.joint_lnp(x.float())

                pad = torch.randn(B, (T-1)*N).to(self.args.device)
                nll = - torch.cat((pad, lnp), dim=1)
                nll = nll.view(B, T,  N) * conf.cuda()
                nll = nll.flatten(start_dim=1)
            
            probs = torch.cat((probs, nll), dim=0) # nll is the anomaly score
        anomaly_scores_kp = probs.cpu().detach().numpy().squeeze().copy(order='C')

        # print(anomaly_scores_kp)
        # print(anomaly_scores_kp.shape)
        # exit()

        auc = score_anomalies(anomaly_scores_kp, self.val_metadata, args=self.args, split='validation')

        print("AUC on val:", auc)
        return auc  
    
    
    def test(self, ret_all=False, sigma=0, conf_weighting=True):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Test")
        for itern, data_arr in enumerate(pbar):

            x = torch.permute(data_arr[0], (0,2,3,1))[..., :2]
            x = x.to(self.args.device, non_blocking=True) 

            conf = torch.permute(data_arr[0], (0,2,3,1))[..., -1]
            if not conf_weighting:
                conf = torch.ones_like(conf)
            B, T, N, _ = x.shape    

            with torch.no_grad():
                lnp =  self.joint_lnp_full_window(x.float())

                pad = torch.randn(B, (T-1)*N).to(self.args.device)
                nll = - torch.cat((pad, lnp), dim=1)
                nll = nll.view(B, T,  N) * conf.cuda()
                nll = nll.flatten(start_dim=1)
                    
            probs = torch.cat((probs, nll), dim=0) # nll is the anomaly score
        anomaly_scores_kp = probs.cpu().detach().numpy().squeeze().copy(order='C')


        auc, frame_scores, frame_gt = score_anomalies(anomaly_scores_kp, self.test_metadata, args=self.args, split='test', sigma=sigma)

        print("AUC on test forward:", auc)
        if ret_all:
            return auc, frame_scores, frame_gt
        else:
            return auc  
        

    


    






