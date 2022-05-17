# common_tools_in_OD

### For the convenience to find tools about Object Detection, I decide to upload my usual tools on github.

### First tools: “check_xml_has_empty.py”

“check_xml_has_empty.py” which is written to find if there is any empty ".xml" file in designative path.


### Second tools: “1VOC划分训练集测试集.py”
"1VOC划分训练集测试集.py", for split Pascal VOC dataset to train_set and val_set.

### Third tools: “2生成图片与xml目录映射文档.py”
"2生成图片与xml目录映射文档.py", for generate image_path-xml_path pair path word string and put them into a 'txt' file, it is obliged, because the need of DataLoader to read image and label to train models on follow-up.

### Tools: “批量更改cocodataset_annotations的image_id.py”
正常从VOC转换到COCO的时候，因为自制VOC格式数据集的图片名字不一定是纯数字。
所以转换之后，image_id有可能不是纯数字，在使用cocoapi做eval的时候可能会出现错误。

因此制作了一个工具将instances_train2017.json或者instances_val2017.json的里面所有的image_id改成仅仅包含数字的字符串，并且将图片名称更改为对应的image_id数字名称

该工具拥有3个功能：
1. 将instances_train2017.json或者instances_val2017.json的里面所有的image_id改成仅仅包含数字的字符串
2. 更改图片的名字，将名字改成"image_id” + “.jpg”，例如"12_3245_23.jpg"改成"42.jpg"(数字不一定是42，是根据该图片名称在json文件里面的记录顺序而定的)
3. 因为考虑到该数据集是从VOC转换过来，除了第2点提到改.jpg图片名称外，还增加了改VOC的Annotations文件夹里面的xml文件名称
