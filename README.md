# DL_Assigment

**Instruction for infer:**

 1. git clone https://github.com/Lamanh1234/DL_Assigment.git

2. Download checkpoint model from drive: https://drive.google.com/file/d/1WW55yvoKnh7QFHalj87GWl39is_9f1zj/view?usp=sharing

3. Put the checkpoint model **model.pth** into folder where you clone above (folder **DL_Assignment**)

4. Because i use some libraries/modulus, you may need to install some library below to avoid error:
    - albumentations : pip install albumentations
    - cv2 : pip install opencv-python
    - torch: pip install torch
    - timm: pip install timm
    - numpy and pandas

6. Open cmd: and change dir to DL Assigment folder   **cd ......\DL_Assignment**
7. **python infer.py**

8. Watiting until appear "Done" then get the **output.csv** file in the folder
