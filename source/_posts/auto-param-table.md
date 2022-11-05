---
title: 模型训练结束后自动整理记录各项参数
tags:
  - machine-learning
  - python
date: 2018-05-14 23:57:03
---


### 模型训练完成后，要注意及时记录保存各种参数，网络结构，分类存档以供后续对比出各种结论，但问题是填写一把这个表格太慢了而且太难受了。。

废话不多说，上脚本：

```
def write_summary(net, opt, summary_info):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    prefix   = './source/summaries/'+net.model_name
    if not os.path.exists(prefix): os.mkdir(prefix)
    sum_path = prefix + '/MiracleYoo_'+current_time+'_'+net.model_name+'_Model_Testing_Record_Form.md'
    with codecs.open('./config.py', 'r', encoding='utf-8') as f:
        raw_data = f.readlines()
        configs  = ''
        for line in raw_data:
            if line.strip().startswith('self.'):
                configs += line.strip().strip('self.')+'\n'

    content = '''
# Model Testing Record Form
| Item Name        | Information |
| ---------        | ----------- |
| Model Name       | %s          |
| Tester's Name    | Miracle Yoo |
| Author's Nmae    | Miracle Yoo |
| Test Time        | %s          |
| Test Position    | %s          |
| Training Epoch   | %d          |
| Highest Test Acc | %.4f        |
| Loss of highest Test Acc| %.4f |
| Last epoch test acc   | %.4f   |
| Last epoch test loss  | %.4f   |
| Last epoch train acc  | %.4f   |
| Last epoch train loss | %.4f   |
| Train Dataset Path    | %s     |
| Test Dataset Path     | %s     |
| Class Number     | %d          |
| Framwork         | Pytorch     |
| Basic Method     | Classify    |
| Input Type       | Char        |
| Criterion        | CrossEntropy|
| Optimizer        | %s          |
| Learning Rate    | %.4f        |
| Embedding dimension   | %d     |
| Data Homogenization   | True   |
| Pretreatment|Remove punctuation|
| Other Major Param |            |
| Other Operation   |            |


## Configs
\```
%s
\```

## Net Structure
\```
%s
\```
    '''%(
        net.model_name,
        current_time,
        opt.TEST_POSITION,
        summary_info['total_epoch'],
        summary_info['best_acc'],
        summary_info['best_acc_loss'],
        summary_info['ave_test_acc'],
        summary_info['ave_test_loss'],
        summary_info['ave_train_acc'],
        summary_info['ave_train_loss'],
        os.path.basename(opt.TRAIN_DATASET_PATH),
        os.path.basename(opt.TEST_DATASET_PATH),
        opt.NUM_CLASSES,
        opt.OPTIMIZER,
        opt.LEARNING_RATE,
        opt.EMBEDDING_DIM,

        configs.strip('\n'),
        str(net)
    )
    with codecs.open(sum_path, 'w+', encoding='utf-8') as f:
        f.writelines(content)
```

记着把上面```前面的\去掉食用~
这个表不全，后面会有补充，内容也可以根据你自己模型和项目的具体情况修改。

## 演示效果如下：

# Model Testing Record Form

| Item Name                | Information             |
| ------------------------ | ----------------------- |
| Model Name               | TextCNNInc              |
| Tester's Name            | Miracle Yoo             |
| Author's Nmae            | Miracle Yoo             |
| Test Time                | 2018-05-13_15:24:43     |
| Test Position            | Gangge Server           |
| Training Epoch           | 100                     |
| Highest Test Acc         | 0.7102                  |
| Loss of highest Test Acc | 0.1721                  |
| Last epoch test acc      | 0.6706                  |
| Last epoch test loss     | 0.1721                  |
| Last epoch train acc     | 0.8904                  |
| Last epoch train loss    | 1.2189                  |
| Train Dataset Path       | knowledge&log_data.txt  |
| Test Dataset Path        | yibot_two_year_test.txt |
| Class Number             | 2411                    |
| Framwork                 | Pytorch                 |
| Basic Method             | Classify                |
| Input Type               | Char                    |
| Criterion                | CrossEntropy            |
| Optimizer                | Adam                    |
| Learning Rate            | 0.0010                  |
| Embedding dimension      | 512                     |
| Data Homogenization      | True                    |
| Pretreatment             | Remove punctuation      |
| Other Major Param        |                         |
| Other Operation          |                         |

## Configs

```
USE_CUDA           = torch.cuda.is_available()
RUNNING_ON_SERVER  = False
NET_SAVE_PATH      = "./source/trained_net/"
TRAIN_DATASET_PATH = "../database/test_train/knowledge&log_data.txt"
TEST_DATASET_PATH  = "../database/test_train/yibot_two_year_test.txt"
NUM_EPOCHS         = 100
BATCH_SIZE         = 8
TOP_NUM            = 4
NUM_WORKERS        = 1
IS_TRAINING        = True
ENSEMBLE_TEST      = False
LEARNING_RATE      = 0.001
RE_TRAIN           = False
TEST_POSITION      = 'Gangge Server'
OPTIMIZER          = 'Adam'
USE_CHAR           = True
USE_WORD2VEC       = True
NUM_CLASSES        = 1890#len(get_labels2idx()[0])
EMBEDDING_DIM      = 512
VOCAB_SIZE         = 20029
CHAR_SIZE          = 3403
LSTM_HID_SIZE      = 512
LSTM_LAYER_NUM     = 2
TITLE_DIM          = 200
SENT_LEN           = 20
LINER_HID_SIZE     = 2000
KERNEL_SIZE        = [1,2,3,4,5]
DILA_TITLE_DIM     = 20
```

## Net Structure

```
TextCNNInc(
  (encoder): Embedding(3394, 512)
  (question_convs): ModuleList(
    (0): Sequential(
      (0): Conv1d(512, 200, kernel_size=(1,), stride=(1,))
      (1): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True)
      (2): ReLU(inplace)
      (3): MaxPool1d(kernel_size=20, stride=20, padding=0, dilation=1, ceil_mode=False)
    )
    (1): Sequential(
      (0): Conv1d(512, 200, kernel_size=(3,), stride=(1,))
      (1): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True)
      (2): ReLU(inplace)
      (3): MaxPool1d(kernel_size=18, stride=18, padding=0, dilation=1, ceil_mode=False)
    )
    (2): Sequential(
      (0): Conv1d(512, 200, kernel_size=(1,), stride=(1,))
      (1): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True)
      (2): ReLU(inplace)
      (3): Conv1d(200, 200, kernel_size=(3,), stride=(1,))
      (4): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True)
      (5): ReLU(inplace)
      (6): MaxPool1d(kernel_size=18, stride=18, padding=0, dilation=1, ceil_mode=False)
    )
    (3): Sequential(
      (0): Conv1d(512, 200, kernel_size=(3,), stride=(1,))
      (1): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True)
      (2): ReLU(inplace)
      (3): Conv1d(200, 200, kernel_size=(5,), stride=(1,))
      (4): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True)
      (5): ReLU(inplace)
      (6): MaxPool1d(kernel_size=14, stride=14, padding=0, dilation=1, ceil_mode=False)
    )
  )
  (fc): Sequential(
    (0): Linear(in_features=800, out_features=2000, bias=True)
    (1): BatchNorm1d(2000, eps=1e-05, momentum=0.1, affine=True)
    (2): ReLU(inplace)
    (3): Dropout(p=0.5)
    (4): Linear(in_features=2000, out_features=2411, bias=True)
  )
)
```

喵喵喵~