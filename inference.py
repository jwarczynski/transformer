import torch


def translate(model, max_output_length, tokenizer, en_enstence, device):
    model.eval()

    encoder_input_ids = tokenizer(en_enstence, return_tensors='pt').input_ids.to(device)
    decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]]).to(device)

    # ecnoder_output = model.encoder(encoder_input_ids)
    for i in range(max_output_length):
        # logits = model.lm_head(
        #     model.l_norm(model.decoder(ecnoder_output, decoder_input_ids))
        # )
        logits = model(encoder_input_ids, decoder_input_ids)
        token = logits[:, -1, ].argmax(-1).item()
        decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[token]]).to(device)], dim=-1)
        if token == tokenizer.sep_token_id:
            break

    return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

# TODO: Add the beam search
