The CV is set up such that scans from the same patient are in the same training or validation set and that the landmark distributions are approximately the same for every set. (see `landmark_distributions.pdf`)
The landmark files are generated with the script `../other/preprocess_landmarks.py` and setting `verse2020 = True`. The landmark file generated from this script is under `landmarks.csv`. If you decide to run the script yourself, due to some naming inconsistencies, you also need to rename the file `GL124_CT_ax-iso-ctd.json` to `GL124_CT-ax-iso-ctd.json` as well as `GL240_CT_ax-iso-ctd.json` to `GL240_CT-ax-iso-ctd.json`.


> 【译】CV的设置使得来自同一患者的扫描在同一训练或验证集中，并且每个集合的里程碑分布大致相同。（请参阅`landmark_distributions.pdf `）地标文件是使用脚本`..生成的/other/preprocess_landmarks.py`和设置`verse2020=True`。从该脚本生成的里程碑文件位于“landmarks.csv”下。如果您决定自己运行该脚本，由于某些命名不一致，您还需要将文件“GL124_CT_ax-iso-ctd.json”重命名为“GL124_CT-ax-iso-catd.json”，并将“GL240_CT_ax-i so-ctd.json'”重命名为”GL240_CT-ax-iso-ctd.json“。
