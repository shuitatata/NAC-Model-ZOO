import torch.nn.functional as F
import torch
from torchaudio.transforms import MelSpectrogram
import logging

logger = logging.getLogger(__name__)

# adversarial loss of generator


def G_adversarial_loss(stft_output, stft_output_length, wave_output, wave_output_length):
    '''
    average max(0, 1-D(G(x))) for all time steps and all discriminators

    Args:
        n_discriminator: number of discriminator, including Wave discriminator and STFT_discriminator
        stft_output: output of STFT_discriminator, i.e. D_stft(G(x)), [batch_size, 1, time]
        stft_output_length: length of stft_output, [batch_size]
        wave_output: output of Wave_discriminator, i.e. D_wave(G(x)), list of [batch_size, 1, time]
        wave_output_length: length of wave_output, [n_discriminator, batch_size]

    Returns:
        adversarial loss of generator

    '''
    stft_output = stft_output.squeeze(3)
    logger.debug(f"stft_output: {stft_output.shape}")
    loss_stft = F.relu(1 - stft_output).sum(dim=2)  # [batch_size, 1]
    logger.debug(f"loss_stft: {loss_stft.shape}")
    loss_stft = loss_stft.squeeze(1)  # [batch_size]
    logger.debug(f"loss_stft: {loss_stft.shape}")
    loss_stft = loss_stft / stft_output_length
    logger.debug(f"stft_output_length: {stft_output_length.shape}")
    logger.debug(f"loss_stft: {loss_stft.shape}")

    loss_wave = torch.cat([F.relu(1 - wave_output[i]).sum(dim=2).squeeze(1)/wave_output_length[i]
                          # [batch_size * n_discriminator]
                           for i in range(len(wave_output))], dim=0)
    
    logger.debug(f"loss_stft: {loss_stft.shape}")
    logger.debug(f"loss_wave: {loss_wave.shape}")

    loss = torch.cat([loss_stft, loss_wave], dim=0)

    return loss.mean()


def D_adversarial_loss(stft_output_x, stft_output_x_hat, stft_output_length, wave_output_x, wave_output_x_hat, wave_output_length):
    '''
    Args:
        stft_output_x: output of STFT_discriminator on ground truth, i.e. D_stft(x), [batch_size, 1, time]
        stft_output_x_hat: output of STFT_discriminator on generated audio, i.e. D_stft(G(x)), [batch_size, 1, time]
        stft_output_length: length of stft_output, [batch_size]
        wave_output_x: output of Wave_discriminator on ground truth, i.e. D_wave(x), list of [batch_size, 1, time]
        wave_output_x_hat: output of Wave_discriminator on generated audio, i.e. D_wave(G(x)), list of [batch_size, 1, time]
        wave_output_length: length of wave_output, [n_discriminator, batch_size]

    Returns:
        adversarial loss of discriminator
    '''
    stft_output_x = stft_output_x.squeeze(3)
    stft_output_x_hat = stft_output_x_hat.squeeze(3)
    loss_stft_x_hat = F.relu(
        1 + stft_output_x_hat).sum(dim=2).squeeze(1) / stft_output_length
    loss_stft_x = F.relu(
        1 - stft_output_x).sum(dim=2).squeeze(1) / stft_output_length

    loss_wave_x_hat = torch.cat([F.relu(1 + wave_output_x_hat[i]).sum(dim=2).squeeze(
        1)/wave_output_length[i] for i in range(len(wave_output_x_hat))], dim=0)
    loss_wave_x = torch.cat([F.relu(1 - wave_output_x[i]).sum(dim=2).squeeze(
        1)/wave_output_length[i] for i in range(len(wave_output_x))], dim=0)

    real_loss = torch.cat([loss_stft_x, loss_wave_x], dim=0)
    fake_loss = torch.cat([loss_stft_x_hat, loss_wave_x_hat], dim=0)

    return real_loss.mean() + fake_loss.mean()


def G_feature_loss(stft_outputs_x, stft_outputs_x_hat, stft_output_lengths, wave_outputs_x, wave_outputs_x_hat, wave_output_lengths):
    '''
    Args:
        stft_outputs_x: output of STFT_discriminator's all layers on ground truth, i.e. D_stft(x), list of [batch_size, channels, time, freq]
        stft_outputs_x_hat: output of STFT_discriminator's all layers on generated audio, i.e. D_stft(G(x)), list of [batch_size, channels, time, freq]
        stft_output_lengths: lengths of stft_output's all layers, [num_layers, batch_size]
        wave_outputs_x: output of Wave_discriminator's all layers on ground truth, i.e. D_wave(x), list of lists [batch_size, channels, time, freq]
        wave_outputs_x_hat: output of Wave_discriminator's all layers on generated audio, i.e. D_wave(G(x)), list of lists [batch_size, channels, time, freq]
        wave_output_lengths: lengths of wave_output's all layers, list of [num_layers, batch_size]
    '''

    # STFT Feature Loss
    stft_loss = 0
    for i, (feat_x, feat_G_x) in enumerate(zip(stft_outputs_x, stft_outputs_x_hat)):
        # [batch_size, channels, time, freq] -> [batch_size]
        layer_loss = ((feat_x - feat_G_x).abs().sum(dim=2) /
                      stft_output_lengths[i].view(-1, 1, 1)).sum(dim=-1).sum(dim=-1)
        stft_loss += layer_loss.mean()
    stft_loss = stft_loss / len(stft_outputs_x)

    # WAVE Feature Loss
    wave_loss = 0
    for disc_idx in range(len(wave_outputs_x)):
        disc_loss = 0
        for i, (feat_x, feat_G_x) in enumerate(zip(wave_outputs_x[disc_idx], wave_outputs_x_hat[disc_idx])):
            # [batch_size, channels, time, freq] -> [batch_size]
            layer_loss = (feat_x - feat_G_x).abs().sum(dim=2).sum(dim=-
                                                                  1) / wave_output_lengths[disc_idx][i].view(-1, 1)
            disc_loss += layer_loss.mean()
        wave_loss += disc_loss / len(wave_outputs_x[disc_idx])
    wave_loss = wave_loss / len(wave_outputs_x)

    # 合并损失
    total_loss = stft_loss + wave_loss

    return total_loss


def G_rec_loss(x, x_hat, epsilon=1e-4):
    '''
    Args:
        x: ground truth audio, [batch_size, 1, time]
        x_hat: generated audio, [batch_size, 1, time]
    '''
    x = x.squeeze(1)
    x_hat = x_hat.squeeze(1)
    total_loss = 0
    for i in range(6, 12):
        size = 2**i
        step = size//4  # 使用整数除法
        melspec = MelSpectrogram(
            n_fft=size, hop_length=step, n_mels=8).to(x.device)
        x_melspec = melspec(x)
        x_hat_melspec = melspec(x_hat)
        alpha_s = (size/2)**0.5
        loss = (x_melspec - x_hat_melspec).abs().sum() + alpha_s*(((torch.log(x_melspec.abs() +
                                                                              epsilon)-torch.log(x_hat_melspec.abs()+epsilon))**2).sum(dim=-2)**0.5).sum()
        total_loss += loss
    return total_loss


def G_criterion(
    config,
    x,
    x_hat,
    stft_outputs_x,
    stft_outputs_x_hat,
    stft_output_lengths,
    wave_outputs_x,
    wave_outputs_x_hat,
    wave_output_lengths,
    return_components=False
):
    """
    计算生成器总损失
    
    Args:
        return_components: 如果为True，返回(总损失, 损失组件字典)；否则只返回总损失
    """
    adv_loss = config.lambda_adv * G_adversarial_loss(
        stft_outputs_x[-1], stft_output_lengths[-1], 
        [output[-1] for output in wave_outputs_x], 
        [output_length[-1] for output_length in wave_output_lengths]
    )
    
    feat_loss = config.lambda_feat * G_feature_loss(
        stft_outputs_x, stft_outputs_x_hat, stft_output_lengths, 
        wave_outputs_x, wave_outputs_x_hat, wave_output_lengths
    )
    
    rec_loss = config.lambda_rec * G_rec_loss(x, x_hat)
    
    total_loss = adv_loss + feat_loss + rec_loss
    
    if return_components:
        components = {
            "adversarial_loss": adv_loss,
            "feature_loss": feat_loss,
            "reconstruction_loss": rec_loss,
            "total_loss": total_loss
        }
        return total_loss, components
    else:
        return total_loss


def D_criterion(
        stft_outputs_x,
        stft_outputs_x_hat,
        stft_output_lengths,
        wave_outputs_x,
        wave_outputs_x_hat,
        wave_output_lengths):

    return D_adversarial_loss(stft_outputs_x[-1], stft_outputs_x_hat[-1], stft_output_lengths[-1], [output[-1] for output in wave_outputs_x], [output[-1] for output in wave_outputs_x_hat], [output_length[-1] for output_length in wave_output_lengths])

