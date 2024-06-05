import os
import csv
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmdet.engine.hooks import DetVisualizationHook

@HOOKS.register_module()
class CustomVisualizationHook(DetVisualizationHook):
    def __init__(self, save_path='results.csv', **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Initialize CSV file with headers
        with open(self.save_path, 'w', newline='') as csvfile:
            fieldnames = ['image_id', 'bbox', 'label', 'score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        """Custom implementation of after_val_iter."""
        super().after_val_iter(runner, batch_idx, data_batch, outputs)
        
        print(outputs)
        # Write outputs to CSV
        # with open(self.save_path, 'a', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=['image_id', 'bbox', 'label', 'score'])
        #     for i, output in enumerate(outputs):
        #         img_id = os.path.basename(output.img_path)
        #         bboxes = output.pred_instances.bboxes if hasattr(output.pred_instances, 'bboxes') else []
        #         labels = output.pred_instances.labels if hasattr(output.pred_instances, 'labels') else []
        #         scores = output.pred_instances.scores if hasattr(output.pred_instances, 'scores') else []

        #         for bbox, label, score in zip(bboxes, labels, scores):
        #             writer.writerow({
        #                 'image_id': img_id,
        #                 'bbox': bbox.tolist(),
        #                 'label': label,
        #                 'score': score
        #             })

        # Additional custom logic can be added here
        print(f'Custom Visualization Hook: Processing batch {batch_idx}')
        
    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        """Custom implementation of after_val_iter."""
        super().after_val_iter(runner, batch_idx, data_batch, outputs)
        
        #print(outputs)
        
        # Write outputs to CSV
        with open(self.save_path, 'a', newline='') as csvfile:
            fieldnames = ['image_id', 'bbox', 'label', 'score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if the file is empty
            if os.stat(self.save_path).st_size == 0:
                writer.writeheader()

            for output in outputs:
                img_id = os.path.basename(output.metainfo['img_path'])
                bboxes = output.pred_instances.bboxes.tolist() if output.pred_instances.bboxes.numel() > 0 else []
                labels = output.pred_instances.labels.tolist() if output.pred_instances.labels.numel() > 0 else []
                scores = output.pred_instances.scores.tolist() if output.pred_instances.scores.numel() > 0 else []

                for bbox, label, score in zip(bboxes, labels, scores):
                    writer.writerow({
                        'image_id': img_id,
                        'bbox': bbox,
                        'label': label,
                        'score': score
                    })

        # Additional custom logic can be added here
        print(f'Custom Visualization Hook: Processing batch {batch_idx}')