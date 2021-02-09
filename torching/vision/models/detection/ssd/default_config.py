from yacs.config import CfgNode as CN

cfg = CN()

cfg.model = CN()
cfg.model.name = 'vgg-ssd'
cfg.model.pretrained = False
cfg.model.weights_file = '/abc/def.pth'

cfg.input = CN()
cfg.input.image_size = [300, 300]
cfg.input.mean_value = [0.406, 0.456, 0.485]
cfg.input.std_value = [0.225, 0.225, 0.225]

cfg.input.augment = CN()

cfg.input.augment.random_expand = CN()
cfg.input.augment.random_expand.prob = 0.0
cfg.input.augment.random_expand.max_ratio = 0.0
cfg.input.augment.random_expand.fill_value = 0.0
cfg.input.augment.random_expand.keep_ratio = True

cfg.input.augment.random_crop = CN()
cfg.input.augment.random_crop.prob = 0.0
cfg.input.augment.random_crop.ratios = [0.1, 0.2]
cfg.input.augment.random_crop.fill_value = 0.0

cfg.input.augment.random_rotate = CN()
cfg.input.augment.random_rotate.prob = 0.0
cfg.input.augment.random_rotate.degrees = []
cfg.input.augment.random_rotate.fill_value = 0.0

cfg.input.augment.random_affine = CN()
cfg.input.augment.random_affine.prob = 0.0
cfg.input.augment.random_affine.degrees = []

cfg.input.augment.random_horizontal_flip = CN()
cfg.input.augment.random_horizontal_flip.prob = 0.0

cfg.input.augment.random_vertical_flip = CN()
cfg.input.augment.random_vertical_flip.prob = 0.0

cfg.input.augment.color_jitter = CN()
cfg.input.augment.color_jitter.brightness = 0.0
cfg.input.augment.color_jitter.contrast = 0.0
cfg.input.augment.color_jitter.saturation = 0.0
cfg.input.augment.color_jitter.hue = 0.0

cfg.backbone = CN()
cfg.backbone.arch = 'vgg16'
cfg.backbone.in_channels = 3
cfg.backbone.out_channels = 1024
cfg.batch_norm = False

cfg.pyramid = CN()
cfg.pyramid.layer_cfg = [
    [256, 1, 1],
    [512, 3, 2],
    [128, 1, 1],
    [256, 3, 2],
    [128, 1, 1],
    [256, 3, 1],
    [128, 1, 1],
    [256, 3, 1]
]

cfg.head = CN()
cfg.head.num_classes = 21

cfg.priorbox = CN()
cfg.priorbox.pyramid_sizes = [38, 19, 10, 5, 3, 1]
# cfg.priorbox.pyramid_sizes = [64, 32, 16, 8, 4, 2, 1]
cfg.priorbox.min_scale = 0.1
cfg.priorbox.max_scale = 1.05
cfg.priorbox.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

cfg.multibox = CN()
cfg.multibox.box_cfg = [4, 6, 6, 6, 4, 4]

cfg.multibox_loss = CN()
cfg.multibox_loss.overlap_thresh = 0.5
cfg.multibox_loss.neg_pos_ratio = 3
cfg.multibox_loss.alpha = 0.5

cfg.box_selector = CN()
cfg.box_selector.nms_threshold = 0.2
cfg.box_selector.top_k = 400
cfg.box_selector.confidence_threshold = 0.01
cfg.box_selector.keep_top_k = 200

cfg.train = CN()
cfg.train.epochs = 100
cfg.train.batch_size = 32
cfg.train.steps = (50, 75)
cfg.train.gamma = 0.1
cfg.train.learning_rate = 0.01
cfg.train.cuda = False