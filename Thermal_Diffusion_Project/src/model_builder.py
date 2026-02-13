from diffusers import UNet2DModel, DDPMScheduler

def create_model_and_scheduler(config):
    model = UNet2DModel(
        sample_size=config['data']['image_size'],
        in_channels=config['data']['channels'],
        out_channels=config['data']['channels'],
        layers_per_block=config['model']['num_res_blocks'],
        block_out_channels=tuple(config['model']['model_channels'] * m for m in config['model']['channel_mult']),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    return model, noise_scheduler
