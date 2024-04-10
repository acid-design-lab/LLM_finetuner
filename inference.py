import time
from typing import Any
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
BASE_MODEL_PATH="Mistral model path"
LORA_MODEL_PATH="Your LoRa adapter path"


# Those fns are needed to "cache" prev results and allow early-stop.
def _extract_past_from_model_output(self, outputs, standardize_cache_format: bool = False):
    past_key_values = None
    if "past_key_values" in outputs:
        past_key_values = outputs.past_key_values
    elif "mems" in outputs:
        past_key_values = outputs.mems
    elif "past_buckets_states" in outputs:
        past_key_values = outputs.past_buckets_states
    return past_key_values


def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
) -> dict[str, Any]:
    # update past_key_values
    model_kwargs["past_key_values"] = _extract_past_from_model_output(
        self,
        outputs,
        standardize_cache_format=standardize_cache_format
    )
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

    if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

    return model_kwargs


def generate(model, input_seq: str, max_length: int, stop_seq: str, **model_kwargs) -> (str, float):
    ids = tokenizer.encode(input_seq, return_tensors="pt")
    ids.to('cuda')
    statistics = {
        "step": [],
        "model": [],
        "score": [],
        "validation": []
    }
    batch_size, cur_len = ids.shape
    model_kwargs["cache_position"] = torch.arange(cur_len, device=ids.device)
    while True:
        step_start = time.perf_counter()
        model_inputs = model.prepare_inputs_for_generation(ids, **model_kwargs)
        output = model(**model_inputs, return_dict=True)
        output_logits = output.logits[:, -1, :]
        model_kwargs = _update_model_kwargs_for_generation(
            model,
            output,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        model_generation_end = time.perf_counter()
        statistics["model"].append(model_generation_end - step_start)
        output_scores = torch.softmax(output_logits, dim=-1)
        next_token = torch.argmax(output_scores)
        score_computation_end = time.perf_counter()
        statistics["score"].append(score_computation_end - model_generation_end)
        ids = torch.cat([ids, torch.tensor([[next_token]])], dim=-1)
        decoded_output = tokenizer.decode(ids[0], skip_special_tokens=True)
        if decoded_output.endswith(stop_seq) or len(ids[0]) >= max_length:
            break
        output_validation_end = time.perf_counter()
        statistics["validation"].append(output_validation_end - score_computation_end)
        statistics["step"].append(output_validation_end - step_start)
    return tokenizer.batch_decode(ids), statistics

# Add your input data to this list.
reacts = ["reaction_1", "reaction_2", "reaction_n"]
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16,
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)

for i in reacts:
    start = time.perf_counter()
    res, statistics = generate(model, MESSAGE % (i,), max_length=2048, temperature=0.1, stop_seq="[DONE]")
    # TODO: save results.
    # Remove metrics if it not needed.
    print(*res)
    print("______________________________________\n")
    step_avg = sum(statistics["step"]) / len(statistics["step"])
    print(f"Avg overall step duration: {step_avg * 1000:.2f}ms")
    model_inf = sum(statistics["model"]) / len(statistics["step"])
    print(f"Avg model inference duration: {model_inf * 1000:.2f}ms", f"({model_inf / step_avg * 100:.2f}%)")
    score = sum(statistics["score"]) / len(statistics["step"])
    print(f"Avg score computation duration: {score * 1000:.2f}ms", f"({score / step_avg * 100:.2f}%)")
    valid = sum(statistics["validation"]) / len(statistics["step"])
    print(f"Avg validation duration: {valid * 1000:.2f}ms", f"({valid / step_avg * 100:.2f}%)")
    print(f"Generation time: {time.perf_counter() - start: .2f}s")
    print("______________________________________\n")
