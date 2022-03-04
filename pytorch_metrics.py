import torch
import piq
from perceptual_metrics.distill_policy import create_image_policy


def flatten_image(x):
    return x.reshape(-1, *x.shape[-3:])


def apply_function_metric(fn, vid1, vid2):
    vid1, vid2 = flatten_image(vid1), flatten_image(vid2)
    return torch.mean(fn(vid1, vid2))


def ssim(vid1, vid2):
    return apply_function_metric(piq.ssim, vid1, vid2)


def psnr(vid1, vid2):
    return apply_function_metric(piq.psnr, vid1, vid2)


def tv(vid1):
    vid1 = flatten_image(vid1)
    return torch.mean(piq.total_variation(vid1))


def lpips(lpips, vid1, vid2):
    return apply_function_metric(lambda x, y: lpips(x, y), vid1, vid2)


class PolicyFeatureL2Metric:

    def __init__(self, policy_path):
        self.image_policy = create_image_policy()
        print(f'Loading policy for metric computation from {policy_path}...')
        self.image_policy.load_state_dict(torch.load(policy_path))
        self.image_policy.eval()
        self.activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook

        self.image_policy.fc_layers[0].register_forward_hook(get_activation('fc'))

    def __call__(self, im1, im2):
        """
        :param im1: batch of images to compare
        :param im2: second batch of images to compare
        :return: two scalars: the mean squared error of policy action outputs between im1 and im2, and the
                 L2 norm of differences of intermediate features when computing the actions for im1 and im2.
        """
        im1, im2 = flatten_image(im1), flatten_image(im2)
        im1_actions = self.image_policy(im1)
        im1_activations = self.activation['fc'].clone()
        im2_actions = self.image_policy(im2)
        im2_activations = self.activation['fc'].clone()
        action_mse = torch.linalg.norm(im1_actions - im2_actions, dim=0).mean()
        activation_mse = torch.linalg.norm(im1_activations - im2_activations, dim=0).mean()
        return action_mse, activation_mse



