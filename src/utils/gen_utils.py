"""
@Desc:
@Reference:
- torch.topk
https://pytorch.org/docs/stable/generated/torch.topk.html
- torch.multinomial
https://pytorch.org/docs/stable/generated/torch.multinomial.html
"""

import torch


def gather_nd(x, indices):
    newshape = list(indices.shape[:-1] + x.shape[indices.shape[-1]:]) + [1]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    out = torch.cat([torch.tensor([x.__getitem__(tuple(i))]) for i in indices]).reshape(tuple(newshape))
    return out


def top_p_logits(logits, p, device=None):
    """Nucleus sampling"""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch, _ = logits.size()
    sorted_logits, _ = torch.sort(logits, descending=True, axis=-1)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), axis=-1)
    cumulative_position = torch.sum((cumulative_probs <= p).to(torch.int32), axis=-1) - 1
    indices = torch.stack([
        torch.arange(0, batch).to(device),
        # number of indices to include
        torch.max(cumulative_position, torch.zeros([batch], dtype=cumulative_position.dtype).to(device)),
    ], axis=-1)
    min_values = gather_nd(sorted_logits, indices).to(device)
    return torch.where(
        logits < min_values,
        torch.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(input_ids, model, max_length, top_p=0.9, tokenizer=None,
                    no_sample=False, device=None):
    if not tokenizer:
        raise ModuleNotFoundError("tokenizer needed")
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = input_ids.size()[0]
    decoder_input_ids = torch.tensor([tokenizer.eos_token_id for _ in range(batch_size)])[:, None].to(device)
    for _ in range(max_length):
        outputs = model(input_ids, decoder_input_ids=decoder_input_ids, use_cache=False, return_dict=True)
        logits = outputs["logits"]
        logits = logits[:, -1, :]

        if no_sample:
            probs = torch.softmax(logits, dim=-1)
            prev = torch.topk(probs, 1).indices
        else:
            logits = top_p_logits(logits, p=top_p)
            probs = torch.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, 1)
        decoder_input_ids = torch.cat([decoder_input_ids, prev], 1)
        # early stop
        if prev[:, 0].eq(tokenizer.eos_token_id).sum() == prev.shape[0]:
            break
    return decoder_input_ids


def ids_to_clean_string(token_list, tokenizer, remain_sp_tokens=False):
    real_s = 0
    for index_, token_ in enumerate(token_list):
        if token_ not in [tokenizer.bos_token_id, tokenizer.eos_token_id]:
            real_s = index_
            break
    token_list = token_list[real_s:]
    string = tokenizer.decode(token_list, skip_special_tokens=False)
    # string = string[:string.find(tokenizer.eos_token)].strip()
    if not remain_sp_tokens:  # remove special tokens in output
        for one in tokenizer.all_special_tokens:
            string = string.replace(one, " ")
    string = " ".join([one for one in string.split()])
    return string
