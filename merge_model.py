import os
import shutil

# from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
# from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict


if __name__ == '__main__':
    """
    若lm_head参数为空，则在from_pretrained加载模型时需指定ignore_mismatched_sizes=True
    """
    experiment_name = "bloomz-7b1-mt-sft-gpu8-1e5"
    epoch, global_step = 1, 3853
    filename = experiment_name + "-epoch={epoch:02d}-step={step}".format(epoch=epoch, step=global_step)

    root_dir = f"/HanFei/output/task/{experiment_name}"
    output_dir = f"{root_dir}/models/global_step{global_step}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    huggingface_dir = os.path.join(root_dir, "best_tfmr")
    files = os.listdir(huggingface_dir)
    for file_name in files:
        if file_name == "pytorch_model.bin":
            continue
        src_path = os.path.join(huggingface_dir, file_name)
        dest_path = os.path.join(output_dir, file_name)
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)

    # lightning deepspeed has saved a directory instead of a file
    save_path = f"{root_dir}/checkpoints/{filename}.ckpt"
    output_path = os.path.join(output_dir, "pytorch_model.bin")
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
