# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.

INITIAL_LOG_LOSS_SCALE = 20.0
NUM_CLASSES = 1000

def create_image_model(args):
    from models.unet import UNetModel
    if args.channel_mult == "":
        if args.image_size == 512:
            args.channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif args.image_size == 256:
            args.channel_mult = (1, 1, 2, 2, 4, 4)
        elif args.image_size == 128:
            args.channel_mult = (1, 1, 2, 3, 4)
        elif args.image_size == 64:
            args.channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {args.image_size}")
    else:
        args.channel_mult = tuple(int(ch_mult) for ch_mult in args.channel_mult.split(","))

    attention_ds = []
    for res in args.attention_resolutions.split(","):
        attention_ds.append(args.image_size // int(res))
    args.out_channels = 2 * args.out_channels if args.learn_sigma else args.out_channels
        
    return UNetModel(
        image_size=args.image_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        latent_dim=args.latent_dim,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=args.dropout,
        channel_mult=args.channel_mult,
        num_classes=(NUM_CLASSES if args.class_cond else None),
        use_checkpoint=args.use_checkpoint,
        use_fp16=args.use_fp16,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        resblock_updown=args.resblock_updown,
        use_new_attention_order=args.use_new_attention_order,
    )
    

def create_sr_image_model(args):
    from models.unet import SuperResModel
    if args.channel_mult == "":
        if args.image_size == 512:
            args.channel_mult = (1, 1, 2, 2, 4, 4)
        elif args.image_size == 256:
            args.channel_mult = (1, 1, 2, 2, 4, 4)
        elif args.image_size == 64:
            args.channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {args.image_size}")
    else:
        args.channel_mult = tuple(int(ch_mult) for ch_mult in args.channel_mult.split(","))

    attention_ds = []
    for res in args.attention_resolutions.split(","):
        attention_ds.append(args.image_size // int(res))
    args.out_channels = 2 * args.out_channels if args.learn_sigma else args.out_channels

    return SuperResModel(
        image_size=args.image_size,
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        model_channels=args.model_channels,
        out_channels=args.out_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=args.dropout,
        channel_mult=args.channel_mult,
        num_classes=(NUM_CLASSES if args.class_cond else None),
        use_checkpoint=args.use_checkpoint,
        use_fp16=args.use_fp16,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        resblock_updown=args.resblock_updown,
        use_new_attention_order=args.use_new_attention_order,
    )


def create_triplane_model(args):
    from models.unet import TriplaneUNetModel
    if args.channel_mult == "":
        if args.image_size == 512:
            args.channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif args.image_size == 256:
            args.channel_mult = (1, 1, 2, 2, 4, 4)
        elif args.image_size == 128:
            args.channel_mult = (1, 1, 2, 3, 4)
        elif args.image_size == 64:
            args.channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {args.image_size}")
    else:
        args.channel_mult = tuple(int(ch_mult) for ch_mult in args.channel_mult.split(","))

    attention_ds = []
    threedaware_ds = []
    for res in args.attention_resolutions.split(","):
        attention_ds.append(args.image_size // int(res))
    for res in args.threedaware_resolutions.split(","):
        threedaware_ds.append(args.image_size // int(res))
    args.attention_resolutions = tuple(attention_ds)
    args.threedaware_resolutions = tuple(threedaware_ds)

    args.out_channels = (args.out_channels if not args.learn_sigma else 2 * args.out_channels)
    
    return TriplaneUNetModel(
        threedaware_resolutions=args.threedaware_resolutions,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        in_channels=args.in_channels,
        model_channels=args.model_channels,
        out_channels=args.out_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
        channel_mult=args.channel_mult,
        dims=2,
        num_classes=(NUM_CLASSES if args.class_cond else None),
        use_checkpoint=args.use_checkpoint,
        use_fp16=args.use_fp16,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        resblock_updown=args.resblock_updown,
        use_new_attention_order=args.use_new_attention_order,
    )


def create_gaussian_diffusion(args):
    from diffusion.classifier_free_diffusion import LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
    from diffusion.spaced_diffusion import SpacedDiffusion, space_timesteps
    
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    if args.use_kl:
        loss_type = LossType.RESCALED_KL
    elif args.rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
        
    if not args.timestep_respacing:
        args.timestep_respacing = [args.diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(args.diffusion_steps, args.timestep_respacing),
        betas=betas,
        model_mean_type=(
            ModelMeanType.EPSILON if not args.predict_xstart else ModelMeanType.START_X
        ),
        model_var_type=(
            (
                ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else ModelVarType.FIXED_SMALL
            )
            if not args.learn_sigma
            else ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=args.rescale_timesteps,
        guidance_strength=1.0 if args.train else args.guidance_strength,
        #latent_dim=args.latent_dim
    )


def create_sr_triplane_model(args):
    from models.unet import TriplaneSuperResModel
    if args.channel_mult == "":
        if args.image_size == 512:
            args.channel_mult = (1, 1, 2, 2, 4, 4)
        elif args.image_size == 256:
            args.channel_mult = (1, 1, 2, 2, 4, 4)
        elif args.image_size == 64:
            args.channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {args.image_size}")
    else:
        args.channel_mult = tuple(int(ch_mult) for ch_mult in args.channel_mult.split(","))

    attention_ds = []
    for res in args.attention_resolutions.split(","):
        attention_ds.append(args.image_size // int(res))
    args.out_channels = 2 * args.out_channels if args.learn_sigma else args.out_channels

    return TriplaneSuperResModel(args)


def create_latent_diffusion_model(args):
    from models.latentnet import MLPSkipNet
    
    skip_layers = tuple([int(nl) for nl in args.skip_layers.split(',')])
    return MLPSkipNet(
        num_channels=args.num_channels,
        skip_layers=skip_layers,
        num_hid_channels=args.num_hid_channels,
        num_layers=args.num_layers,
        num_time_emb_channels=args.num_time_emb_channels,
        activation=args.activation,
        use_norm=args.use_norm,
        condition_bias=args.condition_bias,
        dropout=args.dropout,
        last_act=args.last_act,
        num_time_layers=args.num_time_layers,
        time_last_act=args.time_last_act
    )

'''
def create_sr_triplane_model(args):
    _ = args.small_size  # hack to prevent unused variable
    
    from models.unet import SuperResModel
    
    if args.large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif args.large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif args.large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {args.large_size}")

    attention_ds = []
    for res in args.attention_resolutions.split(","):
        attention_ds.append(args.large_size // int(res))

    return SuperResModel(
        threedaware_resolutions=args.threedaware_resolutions,
        latent_dim=args.latent_dim,
        image_size=args.large_size,
        in_channels=3,
        model_channels=args.num_channels,
        out_channels=(3 if not args.learn_sigma else 6),
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=args.dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if args.class_cond else None),
        use_checkpoint=args.use_checkpoint,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        resblock_updown=args.resblock_updown,
        use_fp16=args.use_fp16,
    )
'''


