from extractor import gaitfeat_compare
from loguru import logger

def recognise_feat(probe_feat, gallery_feat):
    """Recognizes  the features between probe and gallery

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
    Returns:
        pgdict (dict): The id of probe corresponds to the id of gallery
    """
    logger.info("begin recognising")
    pgdict = gaitfeat_compare(probe_feat, gallery_feat)
    logger.info("recognise Done")
    print("================= probe - gallery ===================")
    print(pgdict)
    return pgdict