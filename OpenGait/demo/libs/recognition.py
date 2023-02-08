from extractor import gaitfeat_compare
from loguru import logger

def recognise_feat(probe_feat, gallery_feat):
    logger.info("begin recognising")
    pgdict = gaitfeat_compare(probe_feat, gallery_feat)
    logger.info("recognise Done")
    print("================= probe - gallery ===================")
    print(pgdict)
    return pgdict