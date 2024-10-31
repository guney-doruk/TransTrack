from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import os, shutil, json

class CocoPerson:
    def __init__(self) -> None:
        pass

    def get_person_imgs(self, data_dir:str, data_type:str, display:bool=False) -> None:
        img_dir = '{}/{}/'.format(data_dir, data_type)
        #img_dir ='{}'.format(data_dir)
        ann_file = '{}annotations/instances_{}.json'.format(data_dir,data_type)
        #ann_file = '{}person_annotations_{}.json'.format(data_dir,data_type)
        coco = COCO(ann_file)
        filterClasses = ['person']

        image_ids = coco.getImgIds()
        image_id = image_ids[0]
        image_info = coco.loadImgs(image_id)

        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)

        #get Person cat ID
        cat_id = coco.getCatIds(catNms=filterClasses)
        print(f'cat_id:{cat_id}')

        img_id = coco.getImgIds(catIds=[cat_id[0]])[215] # change the image
        print(f'imgid:{img_id}')
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id[0]], iscrowd=None)
        print(f'annid:{ann_ids}')

        #get annotations
        anns = coco.loadAnns(ann_ids)

        image_path = coco.loadImgs(img_id)[0]['file_name']

        if display:
            image = plt.imread(img_dir + image_path)
            plt.imshow(image)
            # Display the specified annotations
            coco.showAnns(anns, draw_bbox=True)

            plt.axis('off')
            plt.title('Annotations for Image ID: {}'.format(img_id))
            plt.tight_layout()
            plt.show()

    def get_imgs(self, data_dir:str, data_type:str, display:bool=False) -> None:
        img_dir = '{}/{}/'.format(data_dir, data_type)
        #img_dir ='{}'.format(data_dir)
        ann_file = '{}annotations/instances_{}.json'.format(data_dir,data_type)
        #ann_file = '{}person_annotations_{}.json'.format(data_dir,data_type)
        coco = COCO(ann_file)

        # category_ids = coco.getCatIds()
        # num_categories = len(category_ids)
        # print('number of categories: ',num_categories)
        # for ids in category_ids:
        #     cats = coco.loadCats(ids=ids)
        #     print(cats)



        # Load images for the given ids
        image_ids = coco.getImgIds()
        image_id = image_ids[0]  # Change this line to display a different image
        image_info = coco.loadImgs(image_id)

        # Load annotations for the given ids
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)

        # # Get category ids that satisfy the given filter conditions
        filterClasses = ['person']
        # # Fetch class IDs only corresponding to the filterClasses
        catIds = coco.getCatIds()

        # Get image ID that satisfies the given filter conditions
        imgId = coco.getImgIds(catIds=[catIds[0]])[150]
        print(f'imgid:{imgId}')
        
        ann_ids = coco.getAnnIds(imgIds=[imgId], iscrowd=None)
        print(f'annid:{ann_ids}')
        if display:
            anns = coco.loadAnns(ann_ids)
          

            image_path = coco.loadImgs(imgId)[0]['file_name']
            image = plt.imread(img_dir + image_path)
            plt.imshow(image)
            # Display the specified annotations
            coco.showAnns(anns, draw_bbox=True)

            plt.axis('off')
            plt.title('Annotations for Image ID: {}'.format(imgId))
            plt.tight_layout()
            plt.show()


class CocoPersonExtractor:
    def __init__(self, data_dir, data_type='train2017'):
        self.data_dir = data_dir
        self.data_type = data_type
        self.ann_file = os.path.join(data_dir, 'annotations', f'instances_{data_type}.json')
        self.output_dir = None
        self.cat_ids = None
        self.img_ids = None
        self.coco = None
        self.new_annotations = {"images": [], "annotations": [], "categories": []}

    def extract_person_class(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_coco_data()
        self.filter_person_annotations()
        self.save_annotations()

    def load_coco_data(self):
        self.coco = COCO(self.ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)

    def filter_person_annotations(self):
        for img_id in self.img_ids:
            img = self.coco.loadImgs(img_id)[0]
            shutil.copy(os.path.join(self.data_dir, self.data_type, img['file_name']), self.output_dir)
            self.new_annotations["images"].append(img)
            ann_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=self.cat_ids, iscrowd=None)
            annotations = self.coco.loadAnns(ann_ids)
            ## NOTE: There is no problem until there...
            for ann in annotations:
                # ann['image_id'] = len(self.new_annotations["images"]) - 1
                self.new_annotations["annotations"].append(ann)

        self.new_annotations["categories"].append({'supercategory': 'person', 'id': 1, 'name': 'person'})

    def save_annotations(self):
        with open(os.path.join(self.output_dir, f'person_annotations_{self.data_type}.json'), 'w') as outfile:
            json.dump(self.new_annotations, outfile)


if __name__=="__main__":
    data_dir = '/cta/users/grad4/master/dataset/coco17/'
    output_dir = '/cta/users/grad4/master/dataset/coco17/person/val2017'
    data_type = 'val2017'

    coco_person = CocoPerson()
    coco_person.get_imgs(data_dir=data_dir, data_type=data_type, display=True)

    #extractor = CocoPersonExtractor(data_dir, data_type=data_type)
    #extractor.extract_person_class(output_dir)
