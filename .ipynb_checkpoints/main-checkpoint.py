import yaml
from data.dataset import get_dataloader
from models.GPT_2 import GPT2
from trainers.trainer import train

from transformers import BertTokenizer

def main(config_path="config/default.yaml"):
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Prepare tokenizer (if using a custom or Hugging Face tokenizer)
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # 3. Get dataloader
    train_loader, val_dataloader = get_dataloader(
        text_file="/kaggle/input/conversation/conversationxlsx.xlsx", # Path to your input file
        batch_size=config["train"]["batch_size"]
    )

    # 4. Create model
    gpt2_config = GPT2Config(
        vocab_size=config["model"]["vocab_size"],
        num_blocks=config["model"]["num_blocks"],
        num_heads=config["model"]["num_heads"],
        d_model=config["model"]["d_model"],
        max_len=config["model"]["max_len"]
    )
    model = GPT2(gpt2_config)

    # 5. Training

if __name__ == "__main__":
    main()