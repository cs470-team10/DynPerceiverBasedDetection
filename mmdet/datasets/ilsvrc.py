import os.path as osp
import xml.etree.ElementTree as ET

from mmengine.dist import is_main_process
from mmengine.fileio import get_local_path, list_from_file
from mmengine.utils import ProgressBar

from mmdet.registry import DATASETS
from mmdet.utils.typing_utils import List, Union
from .xml_style import XMLDataset

@DATASETS.register_module()
class ILSVRCDataset(XMLDataset):
    """Dataset for ILSVRC2014."""

    METAINFO = {
        'classes':
        ('n02672831', 'n02691156', 'n02219486', 'n02419796', 'n07739125', 'n02454379', 'n07718747', 'n02764044', 'n02766320', 'n02769748', 'n07693725', 'n02777292', 'n07753592', 'n02786058', 'n02787622', 'n02799071', 'n02802426', 'n02807133', 'n02815834', 'n02131653', 'n02206856', 'n07720875', 'n02828884', 'n02834778', 'n02840245', 'n01503061', 'n02870880', 'n02879718', 'n02883205', 'n02880940', 'n02892767', 'n07880968', 'n02924116', 'n02274259', 'n02437136', 'n02951585', 'n02958343', 'n02970849', 'n02402425', 'n02992211', 'n01784675', 'n03000684', 'n03001627', 'n03017168', 'n03062245', 'n03063338', 'n03085013', 'n03793489', 'n03109150', 'n03128519', 'n03134739', 'n03141823', 'n07718472', 'n03797390', 'n03188531', 'n03196217', 'n03207941', 'n02084071', 'n02121808', 'n02268443', 'n03249569', 'n03255030', 'n03271574', 'n02503517', 'n03314780', 'n07753113', 'n03337140', 'n03991062', 'n03372029', 'n02118333', 'n03394916', 'n01639765', 'n03400231', 'n02510455', 'n01443537', 'n03445777', 'n03445924', 'n07583066', 'n03467517', 'n03483316', 'n03476991', 'n07697100', 'n03481172', 'n02342885', 'n03494278', 'n03495258', 'n03124170', 'n07714571', 'n03513137', 'n02398521', 'n03535780', 'n02374451', 'n07697537', 'n03584254', 'n01990800', 'n01910747', 'n01882714', 'n03633091', 'n02165456', 'n03636649', 'n03642806', 'n07749582', 'n02129165', 'n03676483', 'n01674464', 'n01982650', 'n03710721', 'n03720891', 'n03759954', 'n03761084', 'n03764736', 'n03770439', 'n02484322', 'n03790512', 'n07734744', 'n03804744', 'n03814639', 'n03838899', 'n07747607', 'n02444819', 'n03908618', 'n03908714', 'n03916031', 'n00007846', 'n03928116', 'n07753275', 'n03942813', 'n03950228', 'n07873807', 'n03958227', 'n03961711', 'n07768694', 'n07615774', 'n02346627', 'n03995372', 'n07695742', 'n04004767', 'n04019541', 'n04023962', 'n04026417', 'n02324045', 'n04039381', 'n01495701', 'n02509815', 'n04070727', 'n04074963', 'n04116512', 'n04118538', 'n04118776', 'n04131690', 'n04141076', 'n01770393', 'n04154565', 'n02076196', 'n02411705', 'n04228054', 'n02445715', 'n01944390', 'n01726692', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04256520', 'n04270147', 'n02355227', 'n02317335', 'n04317175', 'n04330267', 'n04332243', 'n07745940', 'n04336792', 'n04356056', 'n04371430', 'n02395003', 'n04376876', 'n04379243', 'n04392985', 'n04409515', 'n01776313', 'n04591157', 'n02129604', 'n04442312', 'n06874185', 'n04468005', 'n04487394', 'n03110669', 'n01662784', 'n03211117', 'n04509417', 'n04517823', 'n04536866', 'n04540053', 'n04542943', 'n04554684', 'n04557648', 'n04530566', 'n02062744', 'n04591713', 'n02391049')
        # palette is a list of color tuples, which is used for visualization.
        # ILSVRC2014에서는 palette값이 필요 없다고 함
    }

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            'classes in `XMLDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)

        # loading process takes around 10 mins
        if is_main_process():
            prog_bar = ProgressBar(len(img_ids))

        for img_id in img_ids:
            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = f'{img_id}.JPEG'
            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)

            if is_main_process():
                prog_bar.update()
        return data_list

    def parse_data_info(self, img_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            img_info (dict): Raw image information, usually it includes
                `img_id`, `file_name`, and `xml_path`.

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        img_id = img_info['img_id']
        xml_path = osp.join(self.data_prefix['img'], 'annotations',
                            f'{img_id}.xml')
        data_info['img_id'] = img_id
        data_info['xml_path'] = xml_path

        # deal with xml file
        with get_local_path(
                xml_path, backend_args=self.backend_args) as local_path:
            raw_ann_info = ET.parse(local_path)
        root = raw_ann_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        folder = root.find('folder').text
        img_path = osp.join(self.data_prefix['img'],
                            img_info['file_name'])
        data_info['img_path'] = img_path

        data_info['height'] = height
        data_info['width'] = width

        # Coordinates are in range [0, width - 1 or height - 1]
        data_info['instances'] = self._parse_instance_info(
            raw_ann_info, minus_one=False)
        print(data_info)
        return data_info