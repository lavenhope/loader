import utils.panda_utils as util

if __name__ == '__main__':
  train_person = 'F:/Code/panda/image_annos/person_bbox_train.json'
  train_vehicle = 'F:/Code/panda/image_annos/vehicle_bbox_train.json'
  train_savepath = 'F:/Code/panda/image_annos/train.json'
  #test_person = 'F:/Code/panda/image_annos/person_bbox_train.json'
  #test_vehicle = 'F:/Code/panda/image_annos/vehicle_bbox_train.json'
  #test_savepath = 'F:/Code/panda/image_annos/train.json'
  imgid = util.generate_coco_anno(train_person, train_vehicle, train_savepath, keywords=None) #transfer ground truth to COCO format
  print(imgid)
  #util.generate_res_from_gt(person, vehicle, savepath, keywords=None)