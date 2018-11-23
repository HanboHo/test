import json
import os
import PIL.Image
import io
from panoptic.utils import dataset_util

IMAGES_JSON_FILE="/mnt/internal/99_Data/08_Mapillary" \
                 "/mapillary-vistas-dataset_public_v1.1" \
                 "/validation/panoptic/panoptic_2018.json"

IMAGES_JSON_FILE2="/mnt/internal/99_Data/08_Mapillary" \
                  "/mapillary-vistas-dataset_public_v1.1" \
                  "/testing/panoptic/panoptic_2018.json"

TEST_IMAGE_DIR = "/mnt/internal/99_Data/08_Mapillary" \
                 "/mapillary-vistas-dataset_public_v1.1" \
                 "/testing/images"

tmp = "/home/chen/segmentationchallengesubmission/validation" \
      "/eval_dgx2_mapillary_panoptic_nokl_02_lowerlr_01_newfusion_01_507212/" \
      "panoptic_validation_AT-Fusion_results_ori.json"
tmp = "/home/chen/segmentationchallengesubmission/validation/eval_dgx2_mapillary_panoptic_nokl_02_lowerlr_01_newfusion_01_507212/" \
      "panoptic_validation_AT-Fusion_results_ori.json"

# tmp = "/home/chen/panoptic.json"

with open(tmp, 'r') as fid:
    data = json.load(fid)

del data['images']
del data['annotations']

images = []
files = [file for file in os.listdir(TEST_IMAGE_DIR) if os.path.isfile(
    os.path.join(TEST_IMAGE_DIR, file))]
for filename_raw in files:
    filename = filename_raw.split('/')[-1]
    filename = filename.split('.')[0]
    image_data = dataset_util.read_data(
        TEST_IMAGE_DIR, filename, dataset_util.FLAGS.image_format)
    image = PIL.Image.open(io.BytesIO(image_data))
    image_height = image.height
    image_width = image.width
    image_info = {'file_name': filename + '.' + dataset_util.FLAGS.image_format,
                  'width': image_width,
                  'height': image_height,
                  'id': filename}
    images.append(image_info)
data['images'] = images

with open(IMAGES_JSON_FILE2, 'w') as fid:
    json.dump(data, fid)

print("Finished creating JSON file...")
