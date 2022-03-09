import torch
import torch.nn as nn
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


class PolicyFeatureL2Metric(nn.Module):

    def __init__(self, policy_path, layer):
        super().__init__()
        self.image_policy = create_image_policy()
        print(f'Loading policy for metric computation from {policy_path}...')
        self.image_policy.load_state_dict(torch.load(policy_path))
        self.image_policy.eval()

        # no need to compute grads
        for param in self.image_policy.parameters():
            param.requires_grad = False

        self.activation = {}

        def get_activation(name):
            def hook(model, input, output):
                device = output.device  # prevent multi-GPU concurrency issues
                self.activation[f'{device}/{name}'] = output
            return hook

        ### Layer should be 'fc*' or 'conv*', which will determine the layer for which the hook is applied
        layer_num = int(layer[-1])
        if 'fc' in layer:
            self.image_policy.fc_layers[layer_num].register_forward_hook(get_activation('feats'))
        elif 'conv' in layer:
            self.image_policy.conv_layers[layer_num].register_forward_hook(get_activation('feats'))

    def get_device(self):
        return self.image_policy.conv_layers[0].weight.device

    def __call__(self, im1, im2):
        """
        :param im1: batch of images to compare
        :param im2: second batch of images to compare
        :return: two scalars: the mean squared error of policy action outputs between im1 and im2, and the
                 L2 norm of differences of intermediate features when computing the actions for im1 and im2.
        """
        im1, im2 = flatten_image(im1), flatten_image(im2)
        im1_actions = self.image_policy(im1)
        im1_activations = self.activation[f'{self.get_device()}/feats'].clone()
        im2_actions = self.image_policy(im2)
        im2_activations = self.activation[f'{self.get_device()}/feats'].clone()
        action_mse = torch.linalg.norm(im1_actions - im2_actions, dim=0).mean()
        activation_mse = torch.linalg.norm(im1_activations - im2_activations, dim=0).mean()
        return action_mse, activation_mse


def test_policy_feature_metric():
    policy_metric = PolicyFeatureL2Metric(
        '/viscam/u/stian/perceptual-metrics/perceptual-metrics/perceptual_metrics/distilled_longer.pkl',
        'fc0')
    image_1 = torch.randn(8, 10, 3, 64, 64).cuda()
    image_2 = torch.randn(8, 10, 3, 64, 64).cuda()
    action_mse, feat_mse = policy_metric(image_1, image_2)
    print('action mse', action_mse)
    print('feat mse', feat_mse)


if __name__ == '__main__':
    test_policy_feature_metric()
