import torch, torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb, random
from torchenhanced import Trainer
from .MinGRU import MinGRU
from ...datasets import TokenText
from ...tokenizer import Tokenizer


class MinGRU_Trainer(Trainer):
    def __init__(self, model: MinGRU, train_dataset: TokenText, valid_dataset : TokenText, backwards : bool=True, 
                 detokenizer :Tokenizer=None, optim: Optimizer = None, scheduler: _LRScheduler = None, 
                 state_save_loc=None, device: str = 'cpu',parallel=None, run_name: str = None, project_name: str = None,
                 run_config: dict ={}):
        super().__init__(model, optim, scheduler, state_save_loc=state_save_loc, device=device, parallel=parallel,
                         run_name=run_name, project_name=project_name, run_config=run_config)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # For logging :
        self.batch_loss = []

        self.detokenizer= detokenizer

        self.backwards = backwards # In principle, extractable form dataset, but annoying because I use Subset, so I just give it.

        # Print number of parameters
        print(f"Number of parameters : {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")

    def get_loaders(self,batch_size,num_workers=0):
        t_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        v_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.text_table = wandb.Table(columns=["batches","text"])

        return t_dataloader, v_dataloader

    def process_batch(self, batch_data):
        # Check if correct, but should be :
        loss = self.compute_loss(batch_data)
        
        if(self.do_batch_log) :
            wandb.log({'lr' : self.scheduler.get_last_lr()[0]},commit=False)
    
        return loss

    def compute_loss(self, batch_data):
        token_in, token_truth = batch_data
        B,T = token_truth.shape

        token_in = token_in.to(self.device)
        token_truth = token_truth.to(self.device) # (B, T)

        token_in = self.model.forward(token_in)
        # RESHAPE OTHERWISE, CROSS ENTROPY LOSS BUGS FOR BIG BATCHES
        token_in = token_in.reshape(B*T,-1) # (B*T, C)
        token_truth = token_truth.reshape(B*T)
        # Check if correct, but should be :
        loss = F.cross_entropy(token_in,token_truth,reduction='mean')

        return loss

    def process_batch_valid(self, batch_data):
        # Check if correct, but should be :
        loss = self.compute_loss(batch_data)

        return loss
        

    def valid_log(self):
        #To be implemented when doing validation
        data, _ = self.valid_dataset[random.randint(0,len(self.valid_dataset)-1)] # (T,)*2
        data = data[:5].to(self.device) # only keep first 5 tokens

        phrase_out = self.model.generate(data[None,:],max_new_tokens=100, do_sample=True).cpu() # (1, 5+300)

        if(self.backwards):
            phrase_out= self.detokenizer.detokenize(torch.flip(phrase_out,dims=[1])) # ()
        else :
            phrase_out=self.detokenizer.detokenize(phrase_out)

        self.text_table.add_data(f"{self.steps_done/1000:.1f}k",phrase_out) 
        # Trick to be able to update table on the fly... Fucking wandb
        new_table = wandb.Table(
        columns=self.text_table.columns, data=self.text_table.data
        )
        wandb.log({'gen_samples': new_table},commit=False)