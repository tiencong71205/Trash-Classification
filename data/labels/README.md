## Data Labeling Process

The dataset used in the Trash Classification project was manually labeled using **LabelImg** in YOLO format.
Each image has a corresponding `.txt` label file, where each line describes one object with the following structure:

<class_id> <x_center> <y_center> <width> <height>


In which the coordinate values are normalized according to the image size.

The dataset consists of **3 main labels**:

-  `paper` : 4989 labels
-  `plastic` : 4999 labels
-  `glass` : 5405 labels

This set of labels is used for training models following the **YOLOv5 / YOLOv8** standard.
