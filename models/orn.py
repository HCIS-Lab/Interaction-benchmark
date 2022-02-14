from orn.orn_two_heads.two_heads import orn_two_heads
from backbone import get_maskrcnn_feature_extractor
def get_orn():
	fe = get_maskrcnn_feature_extractor().state_dict()
	model = orn_two_heads(fe)
	return model

model = get_orn()