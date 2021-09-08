import glob
import os
import torch

# ckpt_path = "out/ensemble/mn-bert-base-cased_ensemble_tmp-(3, 3, 3)_sd-34_bs-64_lr-1e-05_dr-0.98_wd-0.0_es-100_me-100-trtid-id_trpid-LAMA_trrid-P1001_vcb-shared/checkpoints/epoch_99_dev_71.8919_test_85.5422.ckpt"
# ckpt_path = "out/ensemble/mn-bert-base-cased_ensemble_tmp-(3, 3, 3)_sd-34_bs-64_lr-1e-05_dr-0.98_wd-0.0_es-100_me-100-trtid-id_trpid-LAMA_trrid-P*_vcb-shared/checkpoints/*.ckpt"
ckpt_path = "out/ensemble/mn-bert-base-cased_ensemble_tmp-(3, 3, 3)_sd-34_bs-64_lr-1e-05_dr-0.98_wd-0.0_es-100_me-100-trtid-id_trpid-LAMA_trrid-P*_vcb-shared/checkpoints"


def entropy(p):
    p = torch.softmax(p, dim=0)
    return -(p * torch.log(p)).sum()

def main():
    uniform_dist = torch.ones(100)
    print(f"H_u: {entropy(uniform_dist)}, mag: {torch.norm(uniform_dist)}")
    for i, ckpt_path_rel in enumerate(glob.glob(ckpt_path)):
        most_recent_ckpt=None
        for ckpt_name in os.listdir(ckpt_path_rel):
            # choose the most recent one I guess
            ckpt = torch.load(os.path.join(ckpt_path_rel, ckpt_name), map_location="cpu")
            most_recent_ckpt = ckpt if (
                    most_recent_ckpt is None or ckpt['time'] > most_recent_ckpt['time']
            ) else most_recent_ckpt
            # ckpt = torch.load(ckpt_filename, map_location="cpu")
        mixture_weights = most_recent_ckpt['embedding']['mixture_weights']
        print(f"{i}) H:   {entropy(mixture_weights)}, mag: {torch.norm(mixture_weights)}")



if __name__ == "__main__":
    main()
