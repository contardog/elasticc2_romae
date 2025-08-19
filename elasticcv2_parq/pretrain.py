from romae.utils import get_encoder_size
from romae.model import RoMAEForPreTraining, RoMAEForPreTrainingConfig, EncoderConfig
from romae.trainer import Trainer, TrainerConfig

from elasticcv2_parq.dataset import ElasticcParquetDataset
from elasticcv2_parq.config import ElasticcConfig


def pretrain(args):
    """
    Pre-training script which will train RoMAForPreTraining on the data.
    """
    config = ElasticcConfig()
    encoder_args = get_encoder_size(config.model_size)

    model_config = RoMAEForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        tubelet_size=(1, 1, 1),
        n_channels=2,
        n_pos_dims=2
    )
    
    if args.lr is not None:
        print("Overridding configured learning rate")
        config.pretrain_lr = args.lr
    if args.batch_size is not None:
        print("Overriding configured batch size")
        config.pretrain_batch_size = args.batch_size
    if args.epochs is not None:
        print("Overridding configured number of epochs")
        config.pretrain_epochs = args.epochs



    model = RoMAEForPreTraining(model_config)
    trainer_config = TrainerConfig(
        warmup_steps=config.pretrain_warmup_steps,
        checkpoint_dir=args.model_name+"_checkpoint_",
        epochs=config.pretrain_epochs,
        base_lr=config.pretrain_lr,
        eval_every=config.pretrain_eval_every,
        save_every=config.pretrain_save_every,
        optimizer_args=config.pretrain_optimargs,
        batch_size= config.pretrain_batch_size,
        project_name= config.project_name + args.model_name,
        entity_name='contardog-university-of-nova-gorica',
        gradient_clip=config.pretrain_grad_clip,
        lr_scaling=True,
        max_checkpoints = 20,
    )
    print("Start pretrain")
    
    trainer = Trainer(trainer_config)
    with (
        ElasticcParquetDataset(args.test_parquet) as test_dataset,
        ElasticcParquetDataset(args.train_parquet,                
                         gaussian_noise=config.gaussian_noise) as train_dataset
    ):
        trainer.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
        )
