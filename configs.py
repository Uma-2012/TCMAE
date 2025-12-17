import ml_collections

def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 18 # Adjusted for 3x3 input
    config.epochs = 200
    config.mask_ratio = 0.5
    config.masking_method = "random_masking"
    config.output_dir = "/home/pguha6/uma/codes/aed_mae_main/checkpoints/Avenue/"
    config.model = "mae_cvt"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train'
    config.resume = False
    
    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "avenue"
    config.avenue_path = "/home/pguha6/uma/dataset/Dataset/CUHK_avenue/Avenue_Dataset(1)/Avenue%20Dataset/"
    config.avenue_gt_path = "/home/pguha6/uma/dataset/Dataset/CUHK_avenue/ground_truth_demo/testing_label_mask/"
    config.input_3d = True # Enables the 3-frame stack logic
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 10
    config.pin_mem = False

    # Unused but kept for compatibility if needed (can be removed from main.py calls too)
    config.grad_weighted_rec_loss = False 
    config.abnormal_score_func = 'L2' # We are using MSE

    return config

def get_configs_shanghai():
    config = ml_collections.ConfigDict()
    config.batch_size = 18 
    config.epochs = 100
    config.mask_ratio = 0.75
    config.masking_method = "random_masking"
    config.output_dir = "/home/pguha6/uma/codes/aed_mae_main/checkpoints/ShanghaiTech/"
    config.model = "mae_cvt"
    config.input_size = (160, 320)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = "train" 
    config.resume= False

    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "shanghai"
    config.shanghai_path = "/home/pguha6/uma/dataset/Dataset/shanghaitech/"
    config.shanghai_gt_path = "/home/pguha6/uma/dataset/Dataset/shanghaitech/testing/test_frame_mask/"
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 10
    config.pin_mem = False
    
    config.grad_weighted_rec_loss = False
    config.abnormal_score_func = 'L2'

    return config