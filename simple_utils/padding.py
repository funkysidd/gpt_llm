import torch


def pad_batch(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    max_item_length = max(len(item) for item in batch)

    inputs = []
    targets = []
    for item in batch:
        input_list = item.copy()

        # `+` simply adds an item to an existing list
        padded_items = max_item_length - len(item)
        input_list += padded_items * [pad_token_id]
        target_list = input_list[1:] + [pad_token_id]

        # Find the offset of the first item added
        offset = max_item_length - padded_items
        target_list[offset:] = (max_item_length - offset) * [ignore_index]

        if allowed_max_length is not None:
            input_list = input_list[:allowed_max_length]
            target_list = target_list[:allowed_max_length]

        # Appends a list to collection of lists
        inputs.append(input_list)
        targets.append(target_list)

    return torch.tensor(inputs), torch.tensor(targets)


if __name__ == "__main__":
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1, inputs_2, inputs_3)

    input_tensors, target_tensors = pad_batch(batch)

    print(input_tensors)
    print(target_tensors)
