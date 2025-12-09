import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from globals import FGSM, PGD, ALPHA, EPSILON, NUM_ITER

def denormalize(batch, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    device = batch.device
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch).to(device)
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def fgsm_attack(image, data_grad, epsilon = 0.25):
    '''
    Implements FGSM attack
    '''
    # epsilon = step size, sign = direction of steepest ascent (UP THE LOSS)
    perturbed_image = image + epsilon*torch.sign(data_grad)
    return perturbed_image


def fgsm_loss(model, criterion, inputs, labels, defense_args, return_preds = True):
    '''
    Implements adversarial loss as defense against FGSM attack
    '''
    alpha = defense_args[ALPHA]
    epsilon = defense_args[EPSILON]
    inputs.requires_grad = True
    # Calculate the loss for the original image
    model.zero_grad()
    out_orig = model(inputs)
    loss_orig = criterion(out_orig, labels)
    loss_orig.backward(retain_graph=True)
    # Calculate the perturbation
    perturb_inputs = fgsm_attack(inputs, inputs.grad, epsilon=epsilon)
    # Calculate the loss for the perturbed image
    model.zero_grad()
    out_perturb = model(perturb_inputs)
    loss_perturb = criterion(out_perturb, labels)
    # Combine the two losses
    loss = (alpha*loss_orig) + ((1-alpha)*loss_perturb)
    if return_preds:
        _, preds = torch.max(out_orig, 1)
        return loss, preds
    else:
        return loss


def pgd_attack(model, data, target, criterion, args):
    '''
    Implements PGD attack
    '''
    alpha = args[ALPHA]
    epsilon = args[EPSILON]
    num_iter = args[NUM_ITER]
    data.requires_grad = True
    # Start with a copy of the data
    orig_data = data.detach().clone()
    # Then iteratively perturb the data in the direction of the gradient
    for iter in range(num_iter):
        model.zero_grad()
        out_i = model(data)
        l_i = criterion(out_i, target)
        l_i.backward()
        perturbed_i = fgsm_attack(data, data.grad, epsilon = alpha)
        # Make sure to clamp the perturbation to the epsilon ball around the original data
        data = torch.clamp(perturbed_i, min = orig_data - epsilon, max = orig_data + epsilon)
        data = data.detach().clone().requires_grad_(True)
    perturbed_data = data
    return perturbed_data

from tqdm import tqdm
def test_attack(model, test_loader, attack_function, attack_args):
    device = 'mps' if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    criterion = nn.CrossEntropyLoss()
    adv_examples = []
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True # Very important for attack!
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 

        # If the initial prediction is wrong, don't attack
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        
        if attack_function == FGSM: 
            # Get the correct gradients wrt the data
            loss.backward()
            # Perturb the data using the FGSM attack
            perturbed_data = fgsm_attack(data, data.grad, epsilon=attack_args[EPSILON])
            # Re-classify the perturbed image
            output = model(perturbed_data)

        elif attack_function == PGD:
            # Get the perturbed data using the PGD attack
            perturbed_data = pgd_attack(model, data, target, criterion, args=attack_args)
            # Re-classify the perturbed image
            output = model(perturbed_data)
        else:
            print(f"Unknown attack {attack_function}")

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] 
        if final_pred.item() == target.item():
            correct += 1
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                original_data = data.squeeze().detach().cpu()
                adv_ex = perturbed_data.squeeze().detach().cpu()
                adv_examples.append( (init_pred.item(), 
                                      final_pred.item(),
                                      denormalize(original_data), 
                                      denormalize(adv_ex)) )

    # Calculate final accuracy
    final_acc = correct/float(len(test_loader))
    print(f"Attack {attack_function}, args: {attack_args}\nTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    return final_acc, adv_examples