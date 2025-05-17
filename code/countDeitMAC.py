import copy
import warnings
import torch
from torchprofile import profile_macs
from util import getMiniTestDataset

warnings.filterwarnings("ignore")
device = torch.device("cuda:2")
images, labels = getMiniTestDataset()
one_batch_images = images.squeeze(1).to(device)

deits = torch.load("./0.9099_deit3_small_patch16_224.pth", map_location=device)
deits.eval()

one_batch_images = images[0].to(device)

mac = profile_macs(deits, one_batch_images)
block_id = 0

@torch.no_grad()
def MACs(model: torch.nn.Module, inputs: torch.Tensor, accu: int=0):
    global mac, block_id
    x = copy.deepcopy(inputs)
    for sub in model.children():
        name = sub._get_name()
        if isinstance(sub, torch.nn.Sequential):
            accu = MACs(sub, x, accu)
        elif name == "Block":
            for attn in sub.children():
                accu += profile_macs(attn, x)
                x = attn(x)
                attn_name = f"{name}.{block_id}.{attn._get_name()}"
                print(f"{attn_name:<20}: {accu / mac:.3f} {accu}")
            block_id += 1
        else:
            accu += profile_macs(sub, x)
            x = sub(x)
            print(f"{name:<20}: {accu / mac:.3f} {accu}")
    return accu

MACs(deits, one_batch_images)