import torch

import utility
import data
import model
import loss
#这个args包含了对应的参数
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)

#../experiment/RCAN_BIX2_G10R20P48/config.txt
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
	#构建Dataloader
    loader = data.Data(args)     #重新构建 loader
	#载入模型
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

