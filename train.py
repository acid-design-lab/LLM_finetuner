import json
import tomli_w
import datasets
import torch
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# This is prompt-phrase. For your task you need to use your own one. Training data will be inserted into `%s`
MESSAGE = """TOPIC HEADER:
  Get all reactions, reagents and conditios from text.
TOPIC TEXT:
  I have the Material Synthesis Technique:
  "%s"
  TASK: Get all reactions with their reagents, solvents and conditions from this Material Synthesis Technique.
  Answer should contain all types of quantities (g, mmol and equiv) for each material. Use "N/A" if no quanities was specified. 
  When you complete this task write "[DONE]".
REPLY:
"""
# path to your hf-dataset (see `Dataset.save_to_disk()`)
DATASET_PATH = "./react-dataset-hf"

# Patch original mistral tokenizer for batch-training
tokenizer = AutoTokenizer.from_pretrained("./mistral_model")
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "./mistral_model",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)


peft_config = LoraConfig(
    lora_dropout=0.1,
    # Rank of individual lora-matrix. Higher value increases precision, locality and training time.
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    # Layers which LoRa will modify:
    # q,k,v-proj - attention projection layers
    # o-proj - projection of attention outputs
    # gate-proj - up-sampling layer before activation in MLP
    # for other model try "all-linear" or inspect your model architecture
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
)

# apply LoRa
model = get_peft_model(model, peft_config)
ds = datasets.load_from_disk(DATASET_PATH)


def tokenize_input(item):
    return tokenizer(MESSAGE % item["input"])


# in this case we use TOML format. Mistral doesn't like jsons.
def tokenize_text(item):
    doc: list = json.loads(item["output"])
    if isinstance(doc, dict):
        doc = [doc]
    doc: str = tomli_w.dumps({"reactions": doc})
    return {
        "text": (MESSAGE % item["input"]) + doc + "\n[DONE]"
    }


ds = ds.map(tokenize_text)

args = TrainingArguments(
    output_dir="./tmp",
    per_device_train_batch_size=2,
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    peft_config=peft_config,
    max_seq_length=None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=args,
    packing=False,
)

trainer.train()
