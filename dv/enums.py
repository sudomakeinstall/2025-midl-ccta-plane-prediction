# System
import enum


class CheckpointEnum(str, enum.Enum):
    last = "last"
    best = "best"
    pretrain = "pretrain"


class TrainingModeEnum(str, enum.Enum):
    trn = "trn"
    val = "val"
    tst = "tst"

    @classmethod
    def inf(cls):
        return {cls.val, cls.tst}


class NormalizationEnum(str, enum.Enum):
    group = "group"
    instance = "instance"
    layer = "layer"
    batch = "batch"


class NonlinearityEnum(str, enum.Enum):
    relu = "relu"
    leaky_relu = "leaky_relu"
    none = "none"


class TransformEnum(str, enum.Enum):
    pca = "pca"
    orthogonal = "orthogonal"
    rigid = "rigid"
