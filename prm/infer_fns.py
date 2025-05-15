import torch
import torch.nn.functional as F

@torch.inference_mode()
def _qwen_math_infer_fn(input_str: str, model, tokenizer, device):
    GOOD_TOKEN = '+'
    BAD_TOKEN = '-'
    STEP_TAG = '\n\n\n\n\n'

    candidate_tokens = tokenizer.encode(f" {GOOD_TOKEN} {BAD_TOKEN}") # [488, 481]
    step_tag_id = torch.tensor([tokenizer.encode(f" {STEP_TAG}")], device=device) # 76325
    input_id = torch.tensor(
        [tokenizer.encode(input_str)], device=device)
    logits = model(input_id).logits[:,:,candidate_tokens]

    scores = logits.softmax(dim=-1)[:,:,0]
    mask = input_id == step_tag_id
    step_scores = scores[mask]
    return step_scores

@torch.inference_mode()
def _qwen_math_prm_infer_fn(input_str, model, tokenizer, device):
    question = input_str.split("\nAnswer:")[0]
    answer = input_str.split("\nAnswer:")[-1]
    messages = [
        {"role": "system", "content": "Answer the question step by step."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    input_ids = tokenizer.encode(
        conversation_str, 
        return_tensors="pt", 
    ).to(model.device)

    outputs = model(input_ids=input_ids)
    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    probabilities = F.softmax(outputs[0], dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    sample = probabilities
    positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
    non_zero_elements_list = positive_probs.cpu().tolist()
    scores = torch.tensor(non_zero_elements_list)
    return scores


@torch.inference_mode()
def _math_shepherd_infer_fn(input_str: str, model, tokenizer, device):
    GOOD_TOKEN = '+'
    BAD_TOKEN = '-'
    STEP_TAG = 'ки'
    candidate_tokens = tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:] # [648, 387]
    step_tag_id = tokenizer.encode(f"{STEP_TAG}")[-1] # 12902

    input_id = torch.tensor(
        [tokenizer.encode(input_str)], device=device)
    logits = model(input_id).logits[:,:,candidate_tokens]
    scores = logits.softmax(dim=-1)[:,:,0] 
    step_scores = scores[input_id == step_tag_id]
    return step_scores